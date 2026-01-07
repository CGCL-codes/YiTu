#ifndef __HOST_DATA_TYPES_H__
#define __HOST_DATA_TYPES_H__

#include <stdint.h>
#include "host_config.h"
#include "xcl2.hpp"

typedef 
struct AccDescriptor {
    cl::CommandQueue q;
    
    std::vector<cl::CommandQueue> big_gs_queue;
    std::vector<cl::CommandQueue> little_gs_queue;
    cl::CommandQueue apply_queue;
    cl::CommandQueue hbm_queue;

    int num_big_krnl = BIG_KERNEL_NUM;
    int num_little_krnl = LITTLE_KERNEL_NUM;

    std::string big_gs_kernel_name = "bigKernelScatterGather";
    std::string little_gs_kernel_name = "littleKernelScatterGather";
    
    std::string hbm_kernel_name = "kernelHBMWrapper";

    std::string apply_kernel_name = "kernelApply";


    std::vector<cl::Kernel> big_gs_krnls;
    std::vector<cl::Kernel> little_gs_krnls;

    cl::Kernel hbm_krnl;
    cl::Event hbm_event;

    cl::Kernel apply_krnl;
    cl::Event apply_event;

    cl::Context context;
    
} acc_descriptor_dt;


typedef 
struct suPartitionDescriptor{
    unsigned int            num_edges;
    unsigned int            num_vertices;

    bool                    kernel_type; // 0 stands big kernel; 1 for little kernel.
    unsigned int            kernel_id;
    
    unsigned int            dst_offset;
    unsigned int            dst_len;

    std::vector<uint, aligned_allocator<uint>> edge_array_host;

    cl::Buffer              edge_array_dev;
    cl_mem_ext_ptr_t        edge_array_ext_ptr;  

    cl::Event               event;    
} subpartition_descriptor_dt;


typedef 
struct PartitionDescriptor{
    unsigned int            num_edges;
    unsigned int            num_vertices;

    bool                    is_dense; // 0 stands big kernel; 1 for little kernel.
    unsigned int            kernel_id;
    
    unsigned int            dst_offset;
    unsigned int            dst_len;

    std::vector<uint, aligned_allocator<uint>> edge_array_host;
    cl::Buffer              edge_array_dev;
    cl_mem_ext_ptr_t        edge_array_ext_ptr;  

    cl::Event               event;    
    uint                    est_cycles;
    
    unsigned int            num_subpartitions = LITTLE_KERNEL_NUM;
    std::vector<subpartition_descriptor_dt> subP;

} partition_descriptor_dt;


typedef 
struct PartitionContainer{
    unsigned int            num_graph_vertices;
    unsigned int            num_graph_edges;

	std::vector<uint,  aligned_allocator<uint>>     vertex_property;
	std::vector<prop_t, aligned_allocator<prop_t>>  edge_property;

    unsigned int            num_partitions;
    std::vector<partition_descriptor_dt> P;

    unsigned int            num_dense_partitions;
    unsigned int            num_sparse_partitions;

    std::vector<partition_descriptor_dt> DP;
    std::vector<partition_descriptor_dt> SP;


    std::vector<cl::Buffer> src_prop_dev;
    std::vector<cl_mem_ext_ptr_t> src_prop_ext_ptr;

    std::vector<uint,  aligned_allocator<uint>>  dst_tmp_prop_host;
    std::vector<cl::Buffer> dst_tmp_prop_dev;
    std::vector<cl_mem_ext_ptr_t> dst_tmp_prop_ext_ptr;

	std::vector<uint,  aligned_allocator<uint>>     outdegree_host;
    cl::Buffer outdegree_dev;
    cl_mem_ext_ptr_t outdegree_ext_ptr;

    cl::Buffer apply_src_prop_dev;
    cl_mem_ext_ptr_t apply_src_prop_ptr;

    
} partition_container_dt;

#endif /* __HOST_DATA_TYPES_H__ */