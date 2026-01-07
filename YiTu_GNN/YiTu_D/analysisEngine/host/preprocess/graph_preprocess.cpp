#include "xcl2.hpp"
#include "host_config.h"
#include "host_data_types.h"
#include "graph_preprocess.h"
#include <numeric>      // std::iota
#include <algorithm>    // std::swap
#include <iomanip>

/*
1.初始化分区容器
2.复制顶点属性和出度信息
3.将边分配到不同分区(基于目标顶点ID)
4.对分区进行填充对齐(8的倍数)
5.设置分区元数据(边数、偏移等)
*/
partition_container_dt partitionGraph (CSR* csr) {

    partition_container_dt partition_container;
    int num_vertex = csr->vertexNum;
    partition_container.num_graph_vertices = num_vertex;
    partition_container.num_graph_edges = csr->edgeNum;

    partition_container.src_prop_dev.resize(NUM_KERNEL);
    partition_container.src_prop_ext_ptr.resize(NUM_KERNEL);

    partition_container.dst_tmp_prop_dev.resize(NUM_KERNEL);
    partition_container.dst_tmp_prop_ext_ptr.resize(NUM_KERNEL);
    partition_container.P.resize(MAX_NUM_PARTITION);

    partition_container.dst_tmp_prop_host.resize(NUM_VERTEX_ALIGNED, 0);

    partition_container.vertex_property.resize(1 * NUM_VERTEX_ALIGNED); //one for
    for(int i = 0; i < num_vertex; i ++)
        partition_container.vertex_property[i] = (csr->vProps[i]); 

    partition_container.outdegree_host.resize(1 * NUM_VERTEX_ALIGNED); //one for
    for(int i = 0; i < num_vertex; i ++)
        partition_container.outdegree_host[i] = csr->rpao[i + 1] - csr->rpao[i]; 

    std::vector<uint> last_src_buffer(MAX_NUM_PARTITION);
    std::fill(last_src_buffer.begin(), last_src_buffer.end(), 0);

    for (int u = 0; u < num_vertex; u++){
        for (int ciao_idx = csr->rpao[u]; ciao_idx < csr->rpao[u + 1]; ciao_idx++) {
            uint src = u; 
            uint dst = csr->ciao[ciao_idx];
            uint part_id = (uint)(dst / PARTITION_SIZE);
            
            // avoid a set of edges across multiple src buffers.
            uint current_src_buffer = floor(src / SRC_BUFFER_SIZE);

            if(current_src_buffer!=last_src_buffer[part_id]){
                int mod8 = (partition_container.P[part_id].edge_array_host.size() / 2) % 8;
                if(mod8){
                    for (int k = 0; k < (8 - mod8); k ++) {
                        uint last_src = partition_container.P[part_id].edge_array_host.end()[-2]; 
                        uint dst = ENDFLAG | 0x80000000; // they won't be processed if the most significant bit is set to 1.
                        partition_container.P[part_id].edge_array_host.insert(partition_container.P[part_id].edge_array_host.end(), {last_src, dst});
                    }
                }
                //std::cout << (partition_container.P[part_id].edge_array_host.size() / 2) << " : " << ((partition_container.P[part_id].edge_array_host.size() / 2) % 8) << std::endl;
                last_src_buffer[part_id] = current_src_buffer;
            }

            partition_container.P[part_id].edge_array_host.insert(partition_container.P[part_id].edge_array_host.end(), {src, dst});
        }
    }

    // We make number of edges of a partition aligned with 8 as the acc can read 8 edges per cycle.
    // we pad a pseudo edge with format of <lastsrc, endflag> to minize the access distance of src and use the endflag
    // as the termination for acc functions. 
    int num_partition = 0;
    for (int part_id = 0; part_id < MAX_NUM_PARTITION; part_id ++){
        if(partition_container.P[part_id].edge_array_host.size() >= 2){
            int current_num_edges = partition_container.P[part_id].edge_array_host.size() / 2;
            uint last_src =  partition_container.P[part_id].edge_array_host.end()[-2]; 
            //uint last_dst =  partition_container.P[part_id].edge_array_host.end()[-1]; 
            for (int k = 0; k < (8 - (current_num_edges % 8)); k ++) {
                // pseudo-edge
                uint src = last_src | 0x80000000; 
                uint dst = ENDFLAG | 0x80000000; // they won't be processed if the most significant bit is set to 1.
                partition_container.P[part_id].edge_array_host.insert(partition_container.P[part_id].edge_array_host.end(), {src, dst});
            }
            //std::cout << (8 - (current_num_edges % 8)) << " edges are padded to the partition " << part_id << std::endl;
            //std::cout << part_id << "th partition edge nume mod 8: "<< partition_container.P[part_id].edge_array_host.size() / 2 % 8 << " . " << std::endl;
            num_partition ++;
            partition_container.P[part_id].num_edges = partition_container.P[part_id].edge_array_host.size() / 2;
            partition_container.P[part_id].dst_offset = part_id * PARTITION_SIZE;
            partition_container.P[part_id].dst_len = PARTITION_SIZE;
        }
    #if 0
        int edges_size_in_MB = partition_container.P[part_id].edge_array_host.size()*sizeof(uint)/1024/1024;
        std::cout << " P" << part_id << " has "<< partition_container.P[part_id].edge_array_host.size()/2 << " edges occupy " << edges_size_in_MB << " MB capacity!" << std::endl;
        if((edges_size_in_MB) > 256){
            std::cout << part_id << " th partition edges exceeds one HBM channel capacity!" <<std::endl;
        }
    #endif
    }

    partition_container.num_partitions = num_partition;
    
    return partition_container;
}

//reorder vertices according to the outdegree of the vertices...
/*
1.计算顶点入度和出度
2.按入度降序排序顶点
3.重建CSR结构保持图连接关系
4.验证重排序的正确性
*/
void reorderGraph(CSR* csr){
    //std::vector<std::vector<uint, aligned_allocator<uint> > > part_edge_array_host(MAX_NUM_PARTITION);
    int num_vertex = csr->vertexNum;
    //std::sort(g.vertices[i]->inVid.begin(), g.vertices[i]->inVid.end());
    std::vector<int> out_degree(num_vertex);
    std::vector<int> in_degree(num_vertex);

    for (int i = 0; i < num_vertex; i ++){
        out_degree[i] = csr->rpao[i + 1] - csr->rpao[i];
        in_degree[i] = csr->rpai[i + 1] - csr->rpai[i];
    }

    //vid_reorder: index: new vertex ID; value: orginal vertex ID
    std::vector<int> vid_reorder(num_vertex);
    std::iota(vid_reorder.begin(), vid_reorder.end(), 0);
    //reorder according "Indegree" and record their indices
    auto comparator = [&in_degree](int a, int b){ return in_degree[a] > in_degree[b]; }; 
    std::sort(vid_reorder.begin(), vid_reorder.end(), comparator);

    DEBUG_PRINTF("[INFO] Balancing edges within partitions...\n");

    // for(int i = 0; i < 100; i ++)
    //     printf("%d ", vid_reorder[i]);
    // for(int pi = 0; pi < MAX_NUM_PARTITION; pi ++){
    //     for(int i = 0; i < PARTITION_SIZE/2; i ++){
    //         if ((pi* PARTITION_SIZE + (PARTITION_SIZE - i) < num_vertex)  ){
    //             int tmp = vid_reorder[pi* PARTITION_SIZE + i];
    //             vid_reorder[pi* PARTITION_SIZE + i] = vid_reorder[pi* PARTITION_SIZE + (PARTITION_SIZE - i)];
    //             vid_reorder[pi* PARTITION_SIZE + (PARTITION_SIZE - i)] = tmp;      
    //         }
    //     }
    // }
    

    std::srand ( unsigned ( std::time(0) ) ); //&& (i % 2 == 1)
    std::size_t shuffle_size = vid_reorder.size() / (MAX_NUM_PARTITION);
    for (int k = 0; k < (MAX_NUM_PARTITION); k ++){
        std::random_shuffle(vid_reorder.begin() + k * shuffle_size, vid_reorder.begin() + (k+1) * shuffle_size);
        //std::random_shuffle(vid_reorder.begin(), vid_reorder.end());
    }


    // printf("\n\n");
    // for(int i = 0; i < 100; i ++)
    //     printf("%d ", vid_reorder[i]);
    //switch index and value: indx: original vertex ID; value: the reordered vertex ID
    std::vector<int> org2new(num_vertex); 
    for (uint i = 0; i < org2new.size(); i++) {
        org2new[vid_reorder[i]] = i;
    }
    
    // do a replication
    std::vector<int> origin_rpao(num_vertex + 1); 
    std::vector<int> origin_rpai(num_vertex + 1); 
    for (uint i = 0; i < origin_rpao.size(); i++){
        origin_rpao[i] = csr->rpao[i];
        origin_rpai[i] = csr->rpai[i];
    }
    std::vector<int> origin_cia(csr->edgeNum);
    for (uint i = 0; i < origin_cia.size(); i++) {
        origin_cia[i] = csr->ciao[i];
    }

    // generate the new CSR...
    csr->rpao[0] = 0;
    csr->rpai[0] = 0;
    int cia_idx = 0;
    for (int i = 0; i < csr->vertexNum; i++) {
        for (int j = origin_rpao[vid_reorder[i]]; j < origin_rpao[vid_reorder[i] + 1]; j ++){
            csr->ciao[cia_idx ++] = org2new[origin_cia[j]];
        }
        csr->rpao[i+1] = csr->rpao[i] + out_degree[vid_reorder[i]];
        csr->rpai[i+1] = csr->rpai[i] + in_degree[vid_reorder[i]];
    }


    //verification of the reordering
    std::vector<int> origin_vertex_prop(num_vertex, 0); 
    std::vector<int> reordered_vertex_prop(num_vertex, 0); 

    for (int i = 0; i < csr->vertexNum; i++) {
        for (int j = origin_rpao[i]; j < origin_rpao[i + 1]; j ++){
            origin_vertex_prop[origin_cia[j]] += 1;
        }
    }

    for (int i = 0; i < csr->vertexNum; i++) {
        for (int j = csr->rpao[i]; j < csr->rpao[i + 1]; j ++){
            reordered_vertex_prop[csr->ciao[j]] += 1;
        }
    }

    for (int i = 0; i < csr->vertexNum; i++) {
        if (reordered_vertex_prop[i] != origin_vertex_prop[vid_reorder[i]])
            std::cout << "mismatch: vertex ID" << i << " orginal value: " \
                << origin_vertex_prop[vid_reorder[i]] << " reordered: " << reordered_vertex_prop[i] << std::endl;
    }
}
