#include "xcl2.hpp"
#include "host_config.h"
#include "host_data_types.h"
#include "partition_schedule.h"
#include "hbm_mapping.h"
#include <numeric>      // std::iota
#include <iomanip>

/*
1.密集分区处理：估计每个分区的计算周期，根据估计周期划分为子分区；
2.稀疏分区处理：合并多个小分区(默认8个)，对合并后的分区进行排序，划分为子分区并估计计算周期
3.内核分配：密集分区分配给Little流水线，稀疏分区分配给Big流水线
*/
int schedulePartitions(partition_container_dt &partition_container){
    
    //1. TODO: find out a suit implementation (number of big kernels and small kernels) as well as number of big or little paritions.

    //2. Spilt dense partitions into sub-partitions; merge sparse partitions to big partitions.   
    //partition_container.num_dense_partitions = partition_container.num_partitions;
    //partition_container.num_dense_partitions = 0; //partition_container.num_partitions; //std::min(2, partition_container.num_partitions);
#if (LITTLE_KERNEL_NUM && BIG_KERNEL_NUM)
    if(partition_container.num_dense_partitions > partition_container.num_partitions) 
        partition_container.num_dense_partitions = partition_container.num_partitions;
#elif LITTLE_KERNEL_NUM
    partition_container.num_dense_partitions = partition_container.num_partitions;
#elif BIG_KERNEL_NUM
    partition_container.num_dense_partitions = 0;
#endif
    
    std::cout << "num_dense_partitions: " << partition_container.num_dense_partitions << std::endl;
    partition_container.DP.resize(partition_container.num_dense_partitions);

    DEBUG_PRINTF("[INFO] Paritioning dense paritions into subparitions...\n");
    for (uint i = 0; i < partition_container.num_dense_partitions; i ++){
        
        partition_container.DP[i].num_subpartitions = LITTLE_KERNEL_NUM; // this number should be the number of big kernels
        partition_container.DP[i].subP.resize(partition_container.DP[i].num_subpartitions);
        
        // ********************* magic estimation logic ! *******************************//
        #define EDGES_PER_CYCLE 8
        #define SRC_PER_CYCLE 10
        #define L_PROFILE_WINDOW 64
        uint edge_access_cycles = partition_container.P[i].edge_array_host.size()/ 2 / EDGES_PER_CYCLE; // in cycles
        uint src_access_cycles = partition_container.num_graph_vertices / SRC_PER_CYCLE; // in cycles
        uint estimated_cycles = 0;
        uint last_window = 0;
        uint last_eid = 0;
        uint last_src = 0;
        std::vector<std::pair<uint, uint>> eid_estcycle_marker;
        for (uint eid = 0; eid < partition_container.P[i].edge_array_host.size()/2; eid ++){
            uint src = partition_container.P[i].edge_array_host[eid * 2] & 0x7fffffff;
            uint window = (int) (eid/L_PROFILE_WINDOW);
            if(window != last_window){
                estimated_cycles += ((src-last_src)/SRC_PER_CYCLE) > (L_PROFILE_WINDOW/EDGES_PER_CYCLE)? 
                                     ((src-last_src)/SRC_PER_CYCLE) : (L_PROFILE_WINDOW/EDGES_PER_CYCLE);
                //estimated_cycles +=    (SRC_BUFFER_SIZE/SRC_PER_CYCLE) + ((eid - last_eid)/EDGES_PER_CYCLE);

                last_window = window;
                last_src = src;
                eid_estcycle_marker.push_back(std::make_pair(estimated_cycles, eid));
            }
        }

        std::cout << "[EST-DENSE] " << std::setw(3) << edge_access_cycles << " edge_access_cycles; " \
                << src_access_cycles << " src_access_cycles " \
                << estimated_cycles << " estimated_cycles " \      
                << estimated_cycles / 200000.0 << " estimated time (ms). " \ 
                << eid_estcycle_marker.size() << " eid_estcycle_marker size. "
                << std::endl;

        partition_container.DP[i].est_cycles = estimated_cycles;
        uint target_cylces_per_subp = estimated_cycles / partition_container.DP[i].num_subpartitions + 1;
        uint last_subpi = 0;
        uint last_eid_marker = 0;
        std::vector<uint> split_range;
        split_range.push_back(last_eid_marker); std::cout << last_eid_marker << "  "; //for the first subparition...
        for (uint k = 0; k < eid_estcycle_marker.size(); k ++){
            uint subpi = eid_estcycle_marker[k].first / target_cylces_per_subp;
            if(subpi != last_subpi){
                last_subpi = subpi;
                last_eid_marker = eid_estcycle_marker[k].second;
                split_range.push_back(last_eid_marker); std::cout << last_eid_marker << "  ";
            }
        }
        split_range.push_back(partition_container.P[i].edge_array_host.size()/2); //for the first subparition...
        
        for (uint subpi = 0; subpi < split_range.size() -1; subpi ++){
            partition_container.DP[i].subP[subpi].edge_array_host.resize(2 * (split_range[subpi+1] - split_range[subpi]));            
            std::copy(
                partition_container.P[i].edge_array_host.begin() + (split_range[subpi] * 2),
                partition_container.P[i].edge_array_host.begin() + (split_range[subpi+1] * 2),
                partition_container.DP[i].subP[subpi].edge_array_host.begin()
            );
        }

        for (uint subpi = 0; subpi < partition_container.DP[i].num_subpartitions; subpi ++){
            //std::cout << i << "th SP " << subpi << "th subp edge num: "<< partition_container.SP[i].subP[subpi].edge_array_host.size() / 2 << " . " << std::endl;
            if(0 == (partition_container.DP[i].subP[subpi].edge_array_host.size() / 2)){
                for(int di = 0; di < 8; di ++) {
                    uint src = partition_container.P[i].edge_array_host.end()[-2] | 0x80000000; 
                    uint dst = ENDFLAG | 0x80000000; // they won't be processed if the most significant bit is set to 1.
                    partition_container.DP[i].subP[subpi].edge_array_host.insert(partition_container.DP[i].subP[subpi].edge_array_host.end(), {src, dst});
                }
            }
        }

        for (uint k = 0; k < partition_container.DP[i].num_subpartitions; k ++){
            partition_container.DP[i].subP[k].num_edges = partition_container.DP[i].subP[k].edge_array_host.size() / 2;
            //std::cout << i << "th DP " << k << "th subp edge nume mod 8: "<< partition_container.DP[i].subP[k].num_edges % 8 << " . " << std::endl;
            partition_container.DP[i].subP[k].dst_offset = partition_container.P[i].dst_offset;
            partition_container.DP[i].subP[k].dst_len = LITTLE_KERNEL_DST_BUFFER_SIZE;
        }
    }

    //**************************************************************************************************************************//
    DEBUG_PRINTF("[INFO] Merging sparse paritions...\n");
    #define MERGE_NUM 8
    partition_container.num_sparse_partitions = partition_container.num_partitions - partition_container.num_dense_partitions;
    partition_container.num_sparse_partitions = ((partition_container.num_sparse_partitions + (MERGE_NUM-1)) / MERGE_NUM);
    partition_container.SP.resize(partition_container.num_sparse_partitions);
    std::cout << "num_sparse_partitions: " << partition_container.num_sparse_partitions << std::endl;
    
    for (uint i = 0; i < partition_container.num_sparse_partitions; i ++){
        
        std::vector<uint> tmp_edge_buffer;
        for(int k = 0; k < MERGE_NUM; k ++){
            uint parti = partition_container.num_dense_partitions + i * MERGE_NUM + k;
            if (parti < partition_container.num_partitions)
                tmp_edge_buffer.insert( tmp_edge_buffer.end(), 
                                        partition_container.P[parti].edge_array_host.begin(), 
                                        partition_container.P[parti].edge_array_host.end()
                                        );
        }
        
        std::vector<std::pair<uint, uint>> edge_pair_array;

        for(uint k = 0; k < tmp_edge_buffer.size()/2; k++){
            edge_pair_array.push_back(std::make_pair(tmp_edge_buffer[2*k], tmp_edge_buffer[2*k + 1]));
        }
        
        // Sort the vector of pairs
        std::sort(std::begin(edge_pair_array), std::end(edge_pair_array), 
                    [&](const auto& a, const auto& b)
                    {return (a.first & (0x80000000 -1)) < (b.first & (0x80000000 -1));}
                );

        for(uint k = 0; k < tmp_edge_buffer.size()/2; k++){
            partition_container.SP[i].edge_array_host.emplace_back(edge_pair_array[k].first);
            //printf("%d \t", edge_pair_array[k].first);
            partition_container.SP[i].edge_array_host.emplace_back(edge_pair_array[k].second);
        }

        partition_container.SP[i].num_edges = partition_container.SP[i].edge_array_host.size() / 2;
        partition_container.SP[i].dst_offset = partition_container.P[partition_container.num_dense_partitions + i * MERGE_NUM].dst_offset;
        partition_container.SP[i].dst_len = BIG_KERNEL_DST_BUFFER_SIZE;
        //std::cout << i << "th SP  edge nume mod 8: "<< partition_container.SP[i].num_edges % 8 << " . " << std::endl;
        //std::cout << partition_container.SP[i].dst_offset << std::endl;
    }

    DEBUG_PRINTF("[INFO] Paritioning sparse paritions into subparitions...\n");
    for (uint i = 0; i < partition_container.num_sparse_partitions; i ++){
        
        partition_container.SP[i].num_subpartitions = BIG_KERNEL_NUM; // this number should be the number of big kernels
        partition_container.SP[i].subP.resize(partition_container.SP[i].num_subpartitions);
        
        // ********************* magic estimation logic ! *******************************//
        #define EDGES_PER_CYCLE 8
        #define MEMORY_REQ_CYCLE 1
        #define PROFILE_WINDOW SRC_BUFFER_SIZE
        uint edge_access_cycles = partition_container.SP[i].edge_array_host.size()/ 2 / EDGES_PER_CYCLE; // in cycles
        uint estimated_cycles = 0;
        uint last_eid = 0;
        std::vector<std::pair<uint, uint>> eid_estcycle_marker;
        uint num_src_memory_request = 0;
        double src_access_cycles = 0; // in cycles
        uint last_src_cacheline = 0;
        uint last_src_set_cacheline = 0;
        uint have_request_flag = 0;
        for (uint eid = 0; eid < partition_container.SP[i].edge_array_host.size()/2; eid ++){
            uint src = partition_container.SP[i].edge_array_host[eid * 2] & 0x7fffffff;
            uint src_cacheline = src >> 4;
            if(src_cacheline != last_src_cacheline) {
                double projected_cycles = (src_cacheline - last_src_cacheline) / 16.0;
                if(projected_cycles < 1) projected_cycles = 1;
                //if(projected_cycles > 10) {std::cout << src_cacheline << " : " << projected_cycles << std::endl;  projected_cycles = 10; }
                src_access_cycles += projected_cycles;
            }
            last_src_cacheline = src_cacheline;
            if((eid % 8) == 7){
                last_src_set_cacheline = src_cacheline;
            };

            if(!(eid % PROFILE_WINDOW)){
                estimated_cycles +=  (src_access_cycles + (PROFILE_WINDOW/EDGES_PER_CYCLE));
                //estimated_cycles += src_access_cycles > (PROFILE_WINDOW/EDGES_PER_CYCLE)? : src_access_cycles, PROFILE_WINDOW/EDGES_PER_CYCLE;
                eid_estcycle_marker.push_back(std::make_pair(estimated_cycles, eid));
                src_access_cycles = 0;
            }
        }
        std::cout << "[EST-SPARSE] " << partition_container.SP[i].edge_array_host.size()/2 << " edges; " \
                << std::setw(3) << edge_access_cycles << " edge_access_cycles; " \
                << estimated_cycles << " estimated_cycles " \      
                << estimated_cycles / 200000.0 << " estimated time (ms). " \  
                << eid_estcycle_marker.size() << " eid_marker_size. "    
                << std::endl;

        partition_container.SP[i].est_cycles = estimated_cycles;
        uint target_cylces_per_subp = estimated_cycles / partition_container.SP[i].num_subpartitions + 1;
        uint last_subpi = 0;
        uint last_eid_marker = 0;
        std::vector<uint> split_range;
        split_range.push_back(0); //for the first subparition...
        for (uint k = 0; k < eid_estcycle_marker.size(); k ++){
            uint subpi = eid_estcycle_marker[k].first / target_cylces_per_subp;
            if(subpi != last_subpi){
                split_range.push_back(eid_estcycle_marker[k].second);
                last_subpi = subpi;
                last_eid_marker = eid_estcycle_marker[k].second;
            }
        }
        split_range.push_back(partition_container.SP[i].edge_array_host.size()/2); //for the first subparition...
                
        for (uint subpi = 0; subpi < split_range.size() - 1; subpi ++){
            partition_container.SP[i].subP[subpi].edge_array_host.resize(2 * (split_range[subpi+1] - split_range[subpi]));            
            std::copy(
                partition_container.SP[i].edge_array_host.begin() + (split_range[subpi] * 2),
                partition_container.SP[i].edge_array_host.begin() + (split_range[subpi+1] * 2),
                partition_container.SP[i].subP[subpi].edge_array_host.begin()
            );
        }


        for (uint subpi = 0; subpi < partition_container.SP[i].num_subpartitions; subpi ++){
            //std::cout << i << "th SP " << subpi << "th subp edge num: "<< partition_container.SP[i].subP[subpi].edge_array_host.size() / 2 << " . " << std::endl;
            if(0 == (partition_container.SP[i].subP[subpi].edge_array_host.size() / 2)){
                for(int di = 0; di < 8; di ++) {
                    uint src = partition_container.SP[i].edge_array_host.end()[-2] | 0x80000000; 
                    uint dst = ENDFLAG | 0x80000000; // they won't be processed if the most significant bit is set to 1.
                    partition_container.SP[i].subP[subpi].edge_array_host.insert(partition_container.SP[i].subP[subpi].edge_array_host.end(), {src, dst});
                }
            }
        }

        for (uint k = 0; k < partition_container.SP[i].num_subpartitions; k ++){
            partition_container.SP[i].subP[k].num_edges = partition_container.SP[i].subP[k].edge_array_host.size() / 2;
            //std::cout << i << "th SP " << k << "th subp edge nume mod 8: "<< partition_container.SP[i].subP[k].num_edges % 8 << " . " << std::endl;
            partition_container.SP[i].subP[k].dst_offset = partition_container.SP[i].dst_offset;
            partition_container.SP[i].subP[k].dst_len = BIG_KERNEL_DST_BUFFER_SIZE;
        }
    }

    //3. schedule dense and sparse paritions to little and big kernels ***************************//
    DEBUG_PRINTF("[INFO] Scheduling dense and sparse paritions to little and big kernels...\n");
    for (uint part_id = 0; part_id < partition_container.num_dense_partitions; part_id ++){
        for(uint subpart_id = 0; subpart_id < partition_container.DP[part_id].num_subpartitions; subpart_id ++){
            partition_container.DP[part_id].subP[subpart_id].kernel_id = (part_id & 0x1)? (LITTLE_KERNEL_NUM - 1- subpart_id) : subpart_id;
        }
    }

    for (uint part_id = 0; part_id < partition_container.num_sparse_partitions; part_id ++){
        for(uint subpart_id = 0; subpart_id < partition_container.SP[part_id].num_subpartitions; subpart_id ++){
            partition_container.SP[part_id].subP[subpart_id].kernel_id = (part_id & 0x1)? (BIG_KERNEL_NUM - 1 - subpart_id) : subpart_id;
            partition_container.SP[part_id].subP[subpart_id].kernel_id += LITTLE_KERNEL_NUM;
            //std::cout << partition_container.SP[part_id].subP[subpart_id].kernel_id << std::endl;
        }
    }
    return 1;
}


// transfer edge lists vertex properties according to the dstination kernel's connectivity.
//将分区数据分配到FPGA的HBM内存（边列表分配，顶点属性传输），管理设备内存缓冲区
int transferPartitions(partition_container_dt &partition_container, acc_descriptor_dt &acc){
    
    cl_int err;
    std::vector<uint> hbm_bank_usage(32, 0);

    DEBUG_PRINTF("Assign partition edgelists to device...\n");
    for (uint part_id = 0; part_id < partition_container.num_dense_partitions; part_id++){

        for(uint subpart_id = 0; subpart_id < partition_container.DP[part_id].num_subpartitions; subpart_id ++){
            partition_container.DP[part_id].subP[subpart_id].edge_array_ext_ptr.obj = partition_container.DP[part_id].subP[subpart_id].edge_array_host.data();
            partition_container.DP[part_id].subP[subpart_id].edge_array_ext_ptr.param = 0;
            int pc_id = 2 * partition_container.DP[part_id].subP[subpart_id].kernel_id;//partition_container.DP[part_id].subP[subpart_id].kernel_id * 2; // each kernel takes two channels at current stage.
            partition_container.DP[part_id].subP[subpart_id].edge_array_ext_ptr.flags = ( pc_id | XCL_MEM_TOPOLOGY);
            //std::cout << pc_id  << " " << std::endl;

            hbm_bank_usage[pc_id] += partition_container.DP[part_id].subP[subpart_id].edge_array_host.size();
        }
    }   
    for (uint part_id = 0; part_id < partition_container.num_sparse_partitions; part_id++){

        for(uint subpart_id = 0; subpart_id < partition_container.SP[part_id].num_subpartitions; subpart_id ++){
            partition_container.SP[part_id].subP[subpart_id].edge_array_ext_ptr.obj = partition_container.SP[part_id].subP[subpart_id].edge_array_host.data();
            partition_container.SP[part_id].subP[subpart_id].edge_array_ext_ptr.param = 0;
            int pc_id = 2 * partition_container.SP[part_id].subP[subpart_id].kernel_id; //big_kernel_hbm_mapping[partition_container.SP[part_id].subP[subpart_id].kernel_id];//partition_container.SP[part_id].subP[subpart_id].kernel_id * 2; // each kernel takes two channels at current stage.
            partition_container.SP[part_id].subP[subpart_id].edge_array_ext_ptr.flags = ( pc_id | XCL_MEM_TOPOLOGY);
            //std::cout << pc_id  << " " << partition_container.SP[part_id].subP[subpart_id].kernel_id << std::endl;
            hbm_bank_usage[pc_id] += partition_container.SP[part_id].subP[subpart_id].edge_array_host.size();
        }
    }   
    // report the usage of memory bank for edge lists....
    for (int i = 0; i < 32; i++) {
        int size_in_MB = hbm_bank_usage[i] * sizeof(uint) / 1024 / 1024;
        std::cout << "[INFO] " << i << "th kernel uses " << size_in_MB << " MB memory for edge list." << std::endl;
        if (size_in_MB >= 256) 
        {
            if (i < (2 * LITTLE_KERNEL_NUM)) 
                exit(-1);
            else 
                return 0;
        }
    }

    for (uint i = 0; i < partition_container.num_dense_partitions; i++) {
        for(uint subpart_id = 0; subpart_id < partition_container.DP[i].num_subpartitions; subpart_id ++){
            OCL_CHECK(err,
                    partition_container.DP[i].subP[subpart_id].edge_array_dev = cl::Buffer(acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                        partition_container.DP[i].subP[subpart_id].edge_array_host.size() * sizeof(uint), &partition_container.DP[i].subP[subpart_id].edge_array_ext_ptr, &err));
            OCL_CHECK(err,
                            err = acc.q.enqueueMigrateMemObjects({partition_container.DP[i].subP[subpart_id].edge_array_dev}, 0 /* 0 means from host*/));            
        }
    }


    for (uint i = 0; i < partition_container.num_sparse_partitions; i++) {
        for(uint subpart_id = 0; subpart_id < partition_container.SP[i].num_subpartitions; subpart_id ++){
            OCL_CHECK(err,
                    partition_container.SP[i].subP[subpart_id].edge_array_dev = cl::Buffer(acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                        partition_container.SP[i].subP[subpart_id].edge_array_host.size() * sizeof(uint), &partition_container.SP[i].subP[subpart_id].edge_array_ext_ptr, &err));
            OCL_CHECK(err,
                            err = acc.q.enqueueMigrateMemObjects({partition_container.SP[i].subP[subpart_id].edge_array_dev}, 0 /* 0 means from host*/));            
        }
    }
    acc.q.finish();

    
    DEBUG_PRINTF("Assign source vertices to device...\n");
    // each kernel keeps one replica of source vertices propertys
    for (int i = 0; i < NUM_KERNEL; i++) {
        partition_container.src_prop_ext_ptr[i].obj = partition_container.vertex_property.data();
        partition_container.src_prop_ext_ptr[i].param = 0;
        int pc_id = apply_kernel_hbm_mapping[i];//mapping[i]; //2 * i + 1; // change it according to the interfaces of kernels. 0, 2, 4, 6, ...
        //if(i >= partition_container.num_dense_partitions) pc_id = 2*i + 1 + 10;
        partition_container.src_prop_ext_ptr[i].flags = (pc_id | XCL_MEM_TOPOLOGY);
    }   
    
    for (int i = 0; i < NUM_KERNEL; i++) {
        OCL_CHECK(err,
                  partition_container.src_prop_dev[i] = cl::Buffer(acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                partition_container.vertex_property.size() * sizeof(uint), &partition_container.src_prop_ext_ptr[i], &err));
    }
    
    // Copy edge lists of partitions to Device Global Memory will be encapsulated in transferDeviceData();
    for (int i = 0; i < NUM_KERNEL; i++) {
        OCL_CHECK(err,
                  err = acc.q.enqueueMigrateMemObjects({partition_container.src_prop_dev[i]}, 0 /* 0 means from host*/));
    }
    acc.q.finish();


    DEBUG_PRINTF("Assign destination vertices to device...\n");
    for (int i = 0; i < NUM_KERNEL; i++) {
        partition_container.dst_tmp_prop_ext_ptr[i].obj = partition_container.dst_tmp_prop_host.data();
        partition_container.dst_tmp_prop_ext_ptr[i].param = 0;
        int pc_id = apply_kernel_hbm_mapping[i];//mapping[i]; //2 * i + 1; // change it according to the interfaces of kernels. 
        partition_container.dst_tmp_prop_ext_ptr[i].flags = (pc_id | XCL_MEM_TOPOLOGY);
    }   

    for (int i = 0; i < NUM_KERNEL; i++) {
        OCL_CHECK(err,
                partition_container.dst_tmp_prop_dev[i] = cl::Buffer(acc.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                partition_container.dst_tmp_prop_host.size() * sizeof(uint), &partition_container.dst_tmp_prop_ext_ptr[i], &err));
    }

    // Copy edge lists of partitions to Device Global Memory will be encapsulated in transferDeviceData();
    for (int i = 0; i < NUM_KERNEL; i++) {
        OCL_CHECK(err,
                  err = acc.q.enqueueMigrateMemObjects({partition_container.dst_tmp_prop_dev[i]}, 0 /* 0 means from host*/));
    }
    acc.q.finish();

    DEBUG_PRINTF("Assign vertex property for the Apply kernel...\n");
    partition_container.apply_src_prop_ptr.obj = partition_container.vertex_property.data();
    partition_container.apply_src_prop_ptr.param = 0;
    partition_container.apply_src_prop_ptr.flags = (30 | XCL_MEM_TOPOLOGY);

    OCL_CHECK(err,
            partition_container.apply_src_prop_dev = cl::Buffer(acc.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                partition_container.vertex_property.size() * sizeof(uint), &partition_container.apply_src_prop_ptr, &err));
    OCL_CHECK(err,
                err = acc.q.enqueueMigrateMemObjects({partition_container.apply_src_prop_dev}, 0 /* 0 means from host*/));
    acc.q.finish();

    return 1;
}
