#include "xcl2.hpp"
#include "host_config.h"
#include "host_data_types.h"
#include "graph_preprocess.h"
#include "fpga_application.h"
#include <numeric>      // std::iota
#include <algorithm>    // std::swap
#include <iomanip>

//执行多层次的结果验证，确保FPGA加速器的计算结果与软件参考实现一致。
int verifyResults(partition_container_dt &partition_container, acc_descriptor_dt &acc, CSR* csr){
    
    cl_int err;
    uint num_vertex = partition_container.num_graph_vertices;
    uint num_edge = partition_container.num_graph_edges;

    //software results....
    std::cout << "Preparing paritions on software results..." << std::endl;
    std::vector<uint, aligned_allocator<uint> > dst_tmp_prop_verfication(NUM_VERTEX_ALIGNED, 0);
    std::fill(dst_tmp_prop_verfication.begin(), dst_tmp_prop_verfication.end(), 0);
    uint processed_edges = 0;
    for (uint part_id = 0; part_id < partition_container.num_dense_partitions; part_id ++){
        for(uint subpart_id = 0; subpart_id < partition_container.DP[part_id].num_subpartitions; subpart_id ++){
            processed_edges += (partition_container.DP[part_id].subP[subpart_id].edge_array_host.size()/2);
            for (uint i = 0; i < (partition_container.DP[part_id].subP[subpart_id].edge_array_host.size()/2); i ++){
                uint src = partition_container.DP[part_id].subP[subpart_id].edge_array_host[2*i];
                uint dst = partition_container.DP[part_id].subP[subpart_id].edge_array_host[2*i + 1];
                if (dst != ENDFLAG) {
                    prop_t src_prop = partition_container.vertex_property[src]; 
                    prop_t update = PROP_COMPUTE_STAGE0(src_prop);
                    if (IS_ACTIVE_VERTEX(update)){
                        dst_tmp_prop_verfication[dst] = PROP_COMPUTE_STAGE3(dst_tmp_prop_verfication[dst], update);
                        // dst_tmp_prop_verfication[dst] += partition_container.vertex_property[src];
                        // std::cout << partition_container.vertex_property[src] ;
                    }
                }
            }
        }
    }
    for (uint part_id = 0; part_id < partition_container.num_sparse_partitions; part_id ++){
        for(uint subpart_id = 0; subpart_id < partition_container.SP[part_id].num_subpartitions; subpart_id ++){            
            processed_edges += (partition_container.SP[part_id].subP[subpart_id].edge_array_host.size()/2);
            for (uint i = 0; i < (partition_container.SP[part_id].subP[subpart_id].edge_array_host.size()/2); i ++){
                uint src = partition_container.SP[part_id].subP[subpart_id].edge_array_host[2*i];
                uint dst = partition_container.SP[part_id].subP[subpart_id].edge_array_host[2*i + 1];
                if (dst != ENDFLAG) {
                    prop_t src_prop = partition_container.vertex_property[src]; 
                    prop_t update = PROP_COMPUTE_STAGE0(src_prop);
                    if (IS_ACTIVE_VERTEX(update)){
                        dst_tmp_prop_verfication[dst] = PROP_COMPUTE_STAGE3(dst_tmp_prop_verfication[dst], update);
                        // dst_tmp_prop_verfication[dst] += partition_container.vertex_property[src];
                        // std::cout << partition_container.vertex_property[src] ;
                    }
                }
            }
        }
    }

    std::cout << "Preparing end-to-end on software results..." << std::endl;

    std::vector<uint, aligned_allocator<uint> > dst_tmp_prop_verfication_e2e(NUM_VERTEX_ALIGNED, 0);
    for (int i = 0; i < csr->vertexNum; i++) {
        for (int j = csr->rpao[i]; j < csr->rpao[i + 1]; j ++){
            prop_t src_prop = partition_container.vertex_property[i];
            prop_t update = PROP_COMPUTE_STAGE0(src_prop);
            if (IS_ACTIVE_VERTEX(update)) {
                dst_tmp_prop_verfication_e2e[csr->ciao[j]] = PROP_COMPUTE_STAGE3(dst_tmp_prop_verfication_e2e[csr->ciao[j]], update);
                //dst_tmp_prop_verfication_e2e[csr->ciao[j]] += partition_container.vertex_property[i];
            }
            // dst_tmp_prop_verfication_e2e[csr->ciao[j]] += partition_container.vertex_property[i];
        }
    }

    {
        std::cout << "Verifying end-to-end on software vs. paritions on software..." << std::endl;
        int error_cnt = 0;
        for (int i = 0; i < partition_container.num_graph_vertices; i ++){
            if (dst_tmp_prop_verfication_e2e[i] != dst_tmp_prop_verfication[i]) {
                //if (error_cnt < 50) std::cout << " i = " << i << " error: mismatch on destination vertex " << i << ": host " \
                << dst_tmp_prop_verfication[i] << " device " << partition_container.dst_tmp_prop_host[i] << std::endl;
                error_cnt ++;
            }
        }
        if (error_cnt) std::cout << " The whole execution has " << error_cnt << " errors" << std::endl;
    }   

    // apply - software version

    for (int i = 0; i < partition_container.num_graph_vertices; i ++){
        // int new_score = 0  + ((108 * (int)dst_tmp_prop_verfication_e2e[i]) >> 7); 
		// int tmp = 0; 
        // if(partition_container.vertex_property[i]) 
        //     tmp = (1 << 16 ) / partition_container.vertex_property[i];
        // dst_tmp_prop_verfication[i] = 0 + (new_score * tmp) >> 16; 
        dst_tmp_prop_verfication[i] = applyFunc((prop_t)dst_tmp_prop_verfication_e2e[i], (prop_t)partition_container.vertex_property[i], (prop_t)partition_container.vertex_property[i], 0);
    }

    // OCL_CHECK(err,
    //     err = acc.q.enqueueMigrateMemObjects({partition_container.apply_src_prop_dev}, CL_MIGRATE_MEM_OBJECT_HOST));
    // acc.q.finish();

    // for (int i = 0; i < 100; i ++){
    //     std::cout << partition_container.vertex_property[i] << std::endl;
    // }

    std::cout << "Read accumulated results from device for verification..." << std::endl;
    OCL_CHECK(err,
        err = acc.q.enqueueMigrateMemObjects({partition_container.dst_tmp_prop_dev[0]}, CL_MIGRATE_MEM_OBJECT_HOST));
    acc.q.finish();

    std::cout << "Verifying end-to-end on hardware vs. end-to-end on software..." << std::endl;
    int error_cnt = 0;
    for (int i = 0; i < partition_container.num_graph_vertices; i ++){
        // if (i < 50 || dst_tmp_prop_verfication[i] != 2147483646) std::cout << " i = " << i << " correct: match on destination vertex " << i << ": host " \
        //     << dst_tmp_prop_verfication[i] << " device " << partition_container.dst_tmp_prop_host[i] << std::endl;
        if (partition_container.dst_tmp_prop_host[i] != dst_tmp_prop_verfication[i]) {
            if (error_cnt < 50) std::cout << " i = " << i << " error: mismatch on destination vertex " << i << ": host " \
            << dst_tmp_prop_verfication[i] << " device " << partition_container.dst_tmp_prop_host[i] << std::endl;
            error_cnt ++;
        }
    }
    if (error_cnt) std::cout << "This iteration has " << error_cnt << " errors" << std::endl;
        
    std::cout << "Processed edges: " << processed_edges << "; Graph edges: " << num_edge << std::endl;
    
    return 1;
}
