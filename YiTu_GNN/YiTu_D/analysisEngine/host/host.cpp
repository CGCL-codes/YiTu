#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iomanip>

#include "xcl2.hpp"

#include "graph.h"
#include "graph_loader.h"
#include "host_config.h"
#include "acc_setup.h"
#include "graph_preprocess.h"
#include "partition_schedule.h"
#include "verify.h"


using namespace std;

/*
图加载与CSR构建
顶点重排（可选）
属性初始化
加速器初始化
图分区与调度
分区数据传输至FPGA
启动FPGA执行（scatter-gather-apply）
结果验证与性能分析
资源回收
*/
int main(int argc, char **argv) {
        
    if(argc < 3){
        std::cout << "[usage]: ./host_program hardware_program.xclbin graph_dataset" << std::endl;
        exit(-1);
    }
    std::string xcl_file = argv[1];
    std::string gName = argv[2];

    std::string path_graph_dataset = "";

    if (xcl::is_emulation()) {
        //gName = "pokec"; 
        //gName = "lj1"; 
        //gName = "wiki-talk";
        gName = "./dataset/rmat-19-32.txt";              
        std::cout << gName <<" (a small graph) is selected for faster execution on emulation flow." << std::endl;
    }

    // ******** Create the CSR for the selected graph and destory the raw graph data to release memory *******//
    DEBUG_PRINTF("Create the CSR for the selected graph and destory the raw graph data to release memory...\n");
    Graph* gptr = createGraph(gName, path_graph_dataset);
    
    CSR* csr = createCsr(gptr);
    int num_vertex = csr->vertexNum;    
    int num_edge = csr->edgeNum;

    int edges_size_in_MB = num_edge*8/1024/1024;
    int vertex_size_in_MB = num_vertex*4/1024/1024;
    std::cout << "Edges (assuming 8 bytes per edge) occupy " << edges_size_in_MB << " MB capacity!" << std::endl;
    std::cout << "Vertices (assuming 4 bytes per vertex) occupy " << vertex_size_in_MB << " MB capacity!" << std::endl;
    std::cout << "Combined: " << (edges_size_in_MB+vertex_size_in_MB) << " MB capacity!" << std::endl;
    if((edges_size_in_MB+vertex_size_in_MB)/1024 > 8){
        std::cout << gName << " edges exceeds HBM capacity... exit..." <<std::endl;
        exit(-1);
    }
    //********************************************************************************************************//

    //******************************************** Reorder graph *********************************************//
#if VERTEX_REORDER_ENABLE == 1
    DEBUG_PRINTF("[INFO] Reordering graph...\n");
    reorderGraph(csr);
#endif
    //********************************************************************************************************//

// for(int numD = 0; numD < 40; numD ++){
    //*************************************** initializeProperty *********************************************//
    DEBUG_PRINTF("Initialize vertex property...\n");
    initializeProperty(csr);
    //********************************************************************************************************//

    // ********************* create acceleration context and kernels... **************************************//
    DEBUG_PRINTF("Initialize the accelerator and create kernels...\n");
    acc_descriptor_dt acc = initAccelerator(xcl_file); 
    // *******************************************************************************************************//

    //******************************* Conduct graph partitioning *********************************************//
    DEBUG_PRINTF("Partitioning graph...\n");
    partition_container_dt partition_container = partitionGraph(csr);
    //********************************************************************************************************//
    
    int numD = 1;
        
    if(argc > 3) numD = atoi(argv[3]);
     
    partition_container.num_dense_partitions = numD;
    // if(partition_container.num_dense_partitions > partition_container.num_partitions) return 1;
    //******************************* Schedule Partitions to kernels *****************************************//
    
    schedulePartitions(partition_container);


    uint fit_hbm = transferPartitions(partition_container, acc);
    // if(!fit_hbm) continue;
    //********************************************************************************************************//

    //****************************************** Execution ***************************************************//
    DEBUG_PRINTF("Accelerator starts computation...\n");
    
    cl_int err;
    double kernel_time_in_sec;
    int num_super_step = 1;
    for (int super_step = 0 ; super_step < num_super_step ; super_step ++)
    {
        // enqueue the hbm wrapper kernel 
        int argvi = 0;
        for (int i = 0; i < NUM_KERNEL; i ++){
            OCL_CHECK(err, err = acc.hbm_krnl.setArg(argvi++, partition_container.src_prop_dev[i])); // one kernel has one src_prop array
        }
        for (int i = 0; i < NUM_KERNEL; i ++){
            OCL_CHECK(err, err = acc.hbm_krnl.setArg(argvi++, partition_container.dst_tmp_prop_dev[i])); // one kernel has one src_prop array
        }

        OCL_CHECK(err, err = acc.hbm_krnl.setArg(argvi++, partition_container.num_dense_partitions)); // one kernel has one src_prop array
        OCL_CHECK(err, err = acc.hbm_krnl.setArg(argvi++, partition_container.num_sparse_partitions)); // one kernel has one src_prop array
        // Invoking the kernel
        OCL_CHECK(err, err = acc.hbm_queue.enqueueTask(acc.hbm_krnl, NULL, &acc.hbm_event));                

        // enqueue the apply kernel
        argvi = 0;
        OCL_CHECK(err, err = acc.apply_krnl.setArg(argvi++, partition_container.apply_src_prop_dev)); // one kernel has one src_prop array
        OCL_CHECK(err, err = acc.apply_krnl.setArg(argvi++, partition_container.num_dense_partitions)); // one kernel has one src_prop array
        OCL_CHECK(err, err = acc.apply_krnl.setArg(argvi++, partition_container.num_sparse_partitions)); // one kernel has one src_prop array
        uint reg = 0;
        OCL_CHECK(err, err = acc.apply_krnl.setArg(argvi++, reg)); 
        OCL_CHECK(err, err = acc.apply_queue.enqueueTask(acc.apply_krnl, NULL, &acc.apply_event));                

        // enqueue big and little kernels 
        for (uint i = 0; i < partition_container.num_dense_partitions; i++) {
            for(uint subpart_id = 0; subpart_id < partition_container.DP[i].num_subpartitions; subpart_id ++){
                uint krnl_id = partition_container.DP[i].subP[subpart_id].kernel_id;         
                uint part_edge_num = partition_container.DP[i].subP[subpart_id].num_edges;
                uint part_dst_offset =  partition_container.DP[i].subP[subpart_id].dst_offset;

                OCL_CHECK(err, err = acc.little_gs_krnls[krnl_id].setArg(0, partition_container.DP[i].subP[subpart_id].edge_array_dev));
                OCL_CHECK(err, err = acc.little_gs_krnls[krnl_id].setArg(1, part_edge_num));
                OCL_CHECK(err, err = acc.little_gs_krnls[krnl_id].setArg(2, part_dst_offset));
                // Invoking the kernel
                DEBUG_PRINTF("%dth DP: %dth subP -> CU %d : %d edges (%0.2f%%)...\n", i, subpart_id, krnl_id, part_edge_num, double(part_edge_num)/num_edge*100);
                OCL_CHECK(err, err = acc.little_gs_queue[krnl_id].enqueueTask(acc.little_gs_krnls[krnl_id], NULL, &partition_container.DP[i].subP[subpart_id].event));                
            }
        }

        for (uint i = 0; i < partition_container.num_sparse_partitions; i++) {
            for(uint subpart_id = 0; subpart_id < partition_container.SP[i].num_subpartitions; subpart_id ++){
                uint krnl_id = partition_container.SP[i].subP[subpart_id].kernel_id; //global ID to big kernel ID        
                uint part_edge_num = partition_container.SP[i].subP[subpart_id].num_edges;
                uint part_dst_offset = partition_container.SP[i].subP[subpart_id].dst_offset;
                DEBUG_PRINTF("%dth SP: %dth subP -> CU %d : %d edges (%0.2f%%)...\n", i, subpart_id, krnl_id, part_edge_num, double(part_edge_num)/num_edge*100);
                OCL_CHECK(err, err = acc.big_gs_krnls[krnl_id- LITTLE_KERNEL_NUM].setArg(0, partition_container.SP[i].subP[subpart_id].edge_array_dev));
                OCL_CHECK(err, err = acc.big_gs_krnls[krnl_id- LITTLE_KERNEL_NUM].setArg(1, part_edge_num));
                OCL_CHECK(err, err = acc.big_gs_krnls[krnl_id- LITTLE_KERNEL_NUM].setArg(2, part_dst_offset));
                // Invoking the kernel
                OCL_CHECK(err, err = acc.big_gs_queue[krnl_id - LITTLE_KERNEL_NUM].enqueueTask(acc.big_gs_krnls[krnl_id - LITTLE_KERNEL_NUM], NULL, &partition_container.SP[i].subP[subpart_id].event));                
            }
        }

        DEBUG_PRINTF("Enqueue kernels finished...\n");
        auto kernel_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < LITTLE_KERNEL_NUM; i ++) acc.little_gs_queue[i].finish(); 
        DEBUG_PRINTF("LITTLE_KERNEL finished...\n");
        for (int i = 0; i < BIG_KERNEL_NUM; i ++) acc.big_gs_queue[i].finish();
        DEBUG_PRINTF("BIG_KERNEL finished...\n");
        acc.apply_queue.finish();
        DEBUG_PRINTF("APPLY KERNEL finished...\n");    
        acc.hbm_queue.finish();
        DEBUG_PRINTF("HBM KERNEL finished...\n");

        auto kernel_end = std::chrono::high_resolution_clock::now();
        kernel_time_in_sec  = std::chrono::duration<double>(kernel_end - kernel_start).count();
        // if(super_step == (num_super_step - 1)) 
        //     std::cout << " " << std::left << std::setw(20) << gName << " e2e high_resolution_clock time : " << kernel_time_in_sec * 1000 \
        //     << " ms; Throughput : " << (double)num_edge/ kernel_time_in_sec/1000000.0 << "MTEPS" << std::endl;           
    }
    //********************************************************************************************************//

    // ******************************* profile execution time of each partition ******************************//
    
    std::vector<unsigned long> sum_exe_time_little_kernel(NUM_KERNEL,0);
    for (uint i = 0; i < partition_container.num_dense_partitions; i++){        
        unsigned long longest_exetime = 0; 
        unsigned long sum_exetime = 0; 
        for(uint subpart_id = 0; subpart_id < partition_container.DP[i].num_subpartitions; subpart_id ++){
            unsigned long start, stop;

            OCL_CHECK(err, 
                    err = partition_container.DP[i].subP[subpart_id].event.getProfilingInfo<unsigned long>(
                        CL_PROFILING_COMMAND_START, &start));
            OCL_CHECK(err,
                    err = partition_container.DP[i].subP[subpart_id].event.getProfilingInfo<unsigned long>(
                        CL_PROFILING_COMMAND_END, &stop));
            unsigned long exe_time = stop - start;

            sum_exe_time_little_kernel[partition_container.DP[i].subP[subpart_id].kernel_id] += exe_time;
            
            if(exe_time > longest_exetime) longest_exetime = exe_time;
            sum_exetime += exe_time;

            std::cout << "[PROFILE]" <<  " DP " << std::setw(3) << i << " subP " << subpart_id <<" : "\
            << std::setw(2) << partition_container.DP[i].subP[subpart_id].kernel_id << " krnl id; " \
            << std::setw(8) << (double)partition_container.DP[i].subP[subpart_id].num_edges/exe_time*1000.0 << " MTEPS; "\
            << std::setw(8) << partition_container.DP[i].subP[subpart_id].num_edges << " partition edges;  " \
            << std::endl;             
        }
        std::cout << "[PROFILE]" <<  " DP " << std::setw(3) << i <<" : "\
        << std::setw(3) << (double)sum_exetime/(longest_exetime *  partition_container.DP[i].num_subpartitions) * 100.0 << "% utilization; "\
        << std::endl;    
    }
    unsigned long little_kernel_exe_time = *max_element(sum_exe_time_little_kernel.begin(), sum_exe_time_little_kernel.end());
    std::cout   << "[INFO]" <<  " little kernel: " << " executed dense partitions: " << partition_container.num_dense_partitions <<", overall time " \
                << *max_element(sum_exe_time_little_kernel.begin(), sum_exe_time_little_kernel.end()) / 1000000.0 \
                << " ms; " << std::endl;   


    std::vector<unsigned long> sum_exe_time_big_kernel(NUM_KERNEL,0);
    for (uint i = 0; i < partition_container.num_sparse_partitions; i++){
        unsigned long longest_exetime = 0; 
        unsigned long sum_exetime = 0;
        for(uint subpart_id = 0; subpart_id < partition_container.SP[i].num_subpartitions; subpart_id ++){
            unsigned long start, stop;

            OCL_CHECK(err, 
                    err = partition_container.SP[i].subP[subpart_id].event.getProfilingInfo<unsigned long>(
                        CL_PROFILING_COMMAND_START, &start));
            OCL_CHECK(err,
                    err = partition_container.SP[i].subP[subpart_id].event.getProfilingInfo<unsigned long>(
                        CL_PROFILING_COMMAND_END, &stop));
            unsigned long exe_time = stop - start;

            sum_exe_time_big_kernel[partition_container.SP[i].subP[subpart_id].kernel_id - LITTLE_KERNEL_NUM] += exe_time;
            
            if(exe_time > longest_exetime) longest_exetime = exe_time;
            sum_exetime += exe_time;

            std::cout << "[PROFILE]" <<  " SP " << std::setw(3) << i << " subP " << subpart_id <<" : "\
            << std::setw(2) <<partition_container.SP[i].subP[subpart_id].kernel_id << " krnl id; " \
            << std::setw(8) << (double)partition_container.SP[i].subP[subpart_id].num_edges/exe_time*1000.0 << " MTEPS; "\
            << std::setw(8) << partition_container.SP[i].subP[subpart_id].num_edges << " partition edges;  " \
            << std::endl;             
        }

        std::cout << "[PROFILE]" <<  " SP " << std::setw(3) << i <<" : "\
        << std::setw(3) << (double)sum_exetime/(longest_exetime *  partition_container.SP[i].num_subpartitions) * 100.0 << "% utilization; "\
        << std::endl;    
    }
    unsigned long big_kernel_exe_time = *max_element(sum_exe_time_big_kernel.begin(), sum_exe_time_big_kernel.end());
    std::cout   << "[INFO]" <<  " big kernel " << " executed sparse partitions " << partition_container.num_sparse_partitions <<", overall time " \
                << *max_element(sum_exe_time_big_kernel.begin(), sum_exe_time_big_kernel.end()) / 1000000.0 \
                << " ms; " << std::endl;   
    
    unsigned long event_e2e_exe_time = (big_kernel_exe_time > little_kernel_exe_time)? big_kernel_exe_time : little_kernel_exe_time;
    std::cout << "[INFO] " << std::setw(20) << gName \
        << ",  numD: " << numD \
        << ",  e2e: " << kernel_time_in_sec * 1000.0 \
        << " ms;  Throught: " << (double)num_edge/ kernel_time_in_sec/1000000.0 \
        << " MTEPS : " << std::endl;      
    //********************************************************************************************************//

    //******************************************* verifyResults **********************************************//
    verifyResults(partition_container, acc, csr);
    //********************************************************************************************************//   

    //**************** free resources ************************************************************************//
    
    for (uint i = 0; i < partition_container.num_dense_partitions; i++){
        //partition_container.P[i].edge_array_host.clear();
        for(uint subpart_id = 0; subpart_id < partition_container.DP[i].num_subpartitions; subpart_id ++){
            partition_container.DP[i].subP[subpart_id].edge_array_host.clear();
        }
    }

    for (uint i = 0; i < partition_container.num_sparse_partitions; i++){
        partition_container.SP[i].edge_array_host.clear();
        for(uint subpart_id = 0; subpart_id < partition_container.SP[i].num_subpartitions; subpart_id ++){
            partition_container.SP[i].subP[subpart_id].edge_array_host.clear();
        }    
    }
// }
    //********************************************************************************************************//   

    return 0;
}
