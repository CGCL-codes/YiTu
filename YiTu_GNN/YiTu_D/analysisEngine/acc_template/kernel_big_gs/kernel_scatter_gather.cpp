#include <hls_stream.h>
#include <string.h>

#include "acc_data_types.h"

#include "l1_api.h"
#include "acc_config.h"

#include "acc_scatter.h"
#include "acc_gather.h"

/*
Big流水线的顶层函数，协调整个Scatter-Gather流程
1.从全局内存读取边数据
2.生成内存请求获取源顶点属性
3.Scatter阶段处理
4.通过Omega网络路由更新数据
5.Gather阶段处理
6.写入最终结果
*/
extern "C" {
    void  bigKernelScatterGather(
        edge_burst_dt   *part_edge_array,
        uint             part_edge_num,
        uint             part_dst_offset,        // the offset of destination vertices buffered in current gs kernel. e.g., 0 to 0 + BUFFERED_MAX_NUM_VERTEX
        hls::stream<b_cacheline_request_pkt>    &b_cacheline_request_stm,
        hls::stream<b_cacheline_response_pkt>   &b_cacheline_response_stm,
        hls::stream<b_tmp_prop_pkt>   &b_tmp_prop_stm
    )
    {
        const int stream_depth = 8;
        const int large_stream_depth = 32;

#pragma HLS DATAFLOW

#pragma HLS INTERFACE m_axi port=part_edge_array offset=slave bundle=gmem0 
#pragma HLS INTERFACE s_axilite port=part_edge_array bundle=control

#pragma HLS INTERFACE s_axilite port=part_edge_num        bundle=control
#pragma HLS INTERFACE s_axilite port=part_dst_offset      bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control


        hls::stream<edge_burst_dt>           edge_burst_stm;
#pragma HLS stream variable=edge_burst_stm  depth=large_stream_depth

        hls::stream<src_burst_dt>          src_burst_stm;
#pragma HLS stream variable=src_burst_stm depth=16

        hls::stream<mem_request_burst_dt>       mem_request_burst_stm;
#pragma HLS stream variable=mem_request_burst_stm depth=large_stream_depth

        hls::stream<ap_uint<512>>       spe_cacheline_stm[SCATTER_PE_NUM];
#pragma HLS stream variable=spe_cacheline_stm depth=large_stream_depth

        hls::stream<cacheline_dt>       cacheline_stm;
#pragma HLS stream variable=cacheline_stm depth=large_stream_depth

        hls::stream<cacheline_response_dt>       cacheline_response_stm;
#pragma HLS stream variable=cacheline_response_stm depth=large_stream_depth

        hls::stream<update_set_dt>           update_set_stm;
#pragma HLS stream variable=update_set_stm depth=16

        hls::stream<update_tuple_dt>           update_tuple_stm[NUM_EDGE_PER_BURST];
#pragma HLS stream variable=update_tuple_stm depth=16
#pragma HLS ARRAY_PARTITION variable=update_tuple_stm dim=0 complete

        hls::stream<update_tuple_dt>           update_tuple_stm_l1[NUM_EDGE_PER_BURST];
#pragma HLS stream variable=update_tuple_stm_l1 depth=2
#pragma HLS ARRAY_PARTITION variable=update_tuple_stm_l1 dim=0 complete

        hls::stream<update_tuple_dt>           update_tuple_stm_l2[NUM_EDGE_PER_BURST];
#pragma HLS stream variable=update_tuple_stm_l2 depth=2
#pragma HLS ARRAY_PARTITION variable=update_tuple_stm_l2 dim=0 complete

        hls::stream<update_tuple_dt>           update_tuple_stm_l3[NUM_EDGE_PER_BURST];
#pragma HLS stream variable=update_tuple_stm_l3 depth=2
#pragma HLS ARRAY_PARTITION variable=update_tuple_stm_l3 dim=0 complete


        hls::stream<update_tuple_dt>           reduced_update_tuple_stm[REDUCTIONTREE_NUM];
#pragma HLS stream variable=reduced_update_tuple_stm depth=4

        hls::stream<update_tuple_dt>           update_tuple_stm_gather[GATHER_PE_NUM];
#pragma HLS stream variable=update_tuple_stm_gather depth=4

        hls::stream<ap_uint<64>>       tmp_prop_stm[GATHER_PE_NUM];
#pragma HLS stream variable=tmp_prop_stm depth=4

        // read edges from the global memory
        readEdges:
        for(int i = 0; i < (part_edge_num >> LOG2_NUM_EDGE_PER_BURST); i ++) // 8 edges per read, hence << 3, it needs to be tuned according to the data width of edges.
        {
            edge_burst_dt one_edge_burst = part_edge_array[i];

            for (int u = 0; u < NUM_EDGE_PER_BURST; u ++){
        #pragma HLS UNROLL
                ap_uint<20> dst = one_edge_burst.edges[u].dst.range(30,0) - part_dst_offset;
                dst.range(19,19) = one_edge_burst.edges[u].dst.range(31,31);
                one_edge_burst.edges[u].dst = dst;
            }
            write_to_stream(edge_burst_stm, one_edge_burst);


            src_burst_dt one_src_burst; //duplicate the src to decouple access and execute...
            for (int u = 0; u < NUM_EDGE_PER_BURST; u ++){
        #pragma HLS UNROLL
                one_src_burst.src[u] = one_edge_burst.edges[u].src;
            }
            write_to_stream(src_burst_stm, one_src_burst);
        }
        DEBUG_PRINTF("reading edge exits! with %d edge sets read.\n", (part_edge_num >> LOG2_NUM_EDGE_PER_BURST));

        genMemRequest(src_burst_stm, mem_request_burst_stm, part_edge_num);

        sendCachelineRequest(mem_request_burst_stm, cacheline_stm);
        stream2axistream(cacheline_stm, b_cacheline_request_stm);
        
        axistream2stream(b_cacheline_response_stm, cacheline_response_stm);
        receiveResponses(cacheline_response_stm, spe_cacheline_stm);

        accScatter(spe_cacheline_stm, edge_burst_stm, update_set_stm, part_edge_num);

        dispatchUpdateTuples(update_set_stm, update_tuple_stm, part_edge_num);
        //三层交换网络
        //layer 1
        switch2x2<0>(2, update_tuple_stm[0], update_tuple_stm[4], update_tuple_stm_l1[0], update_tuple_stm_l1[1]);
        switch2x2<1>(2, update_tuple_stm[1], update_tuple_stm[5], update_tuple_stm_l1[2], update_tuple_stm_l1[3]);
        switch2x2<2>(2, update_tuple_stm[2], update_tuple_stm[6], update_tuple_stm_l1[4], update_tuple_stm_l1[5]);
        switch2x2<3>(2, update_tuple_stm[3], update_tuple_stm[7], update_tuple_stm_l1[6], update_tuple_stm_l1[7]);

        //layer 2
        switch2x2<4>(1, update_tuple_stm_l1[0], update_tuple_stm_l1[4], update_tuple_stm_l2[0], update_tuple_stm_l2[1]);
        switch2x2<5>(1, update_tuple_stm_l1[1], update_tuple_stm_l1[5], update_tuple_stm_l2[2], update_tuple_stm_l2[3]);
        switch2x2<6>(1, update_tuple_stm_l1[2], update_tuple_stm_l1[6], update_tuple_stm_l2[4], update_tuple_stm_l2[5]);
        switch2x2<7>(1, update_tuple_stm_l1[3], update_tuple_stm_l1[7], update_tuple_stm_l2[6], update_tuple_stm_l2[7]);

        // layer 3
        switch2x2<8> (0, update_tuple_stm_l2[0], update_tuple_stm_l2[4], update_tuple_stm_l3[0], update_tuple_stm_l3[1]);
        switch2x2<9> (0, update_tuple_stm_l2[1], update_tuple_stm_l2[5], update_tuple_stm_l3[2], update_tuple_stm_l3[3]);
        switch2x2<10>(0, update_tuple_stm_l2[2], update_tuple_stm_l2[6], update_tuple_stm_l3[4], update_tuple_stm_l3[5]);
        switch2x2<11>(0, update_tuple_stm_l2[3], update_tuple_stm_l2[7], update_tuple_stm_l3[6], update_tuple_stm_l3[7]);

        // reductionTree<0> (0, update_tuple_stm_array[0],  reduced_update_tuple_stm[0]);
        // reductionTree<1> (1, update_tuple_stm_array[1],  reduced_update_tuple_stm[1]);
        // reductionTree<2> (2, update_tuple_stm_array[2],  reduced_update_tuple_stm[2]);
        // reductionTree<3> (3, update_tuple_stm_array[3],  reduced_update_tuple_stm[3]);
        // reductionTree<4> (4, update_tuple_stm_array[4],  reduced_update_tuple_stm[4]);
        // reductionTree<5> (5, update_tuple_stm_array[5],  reduced_update_tuple_stm[5]);
        // reductionTree<6> (6, update_tuple_stm_array[6],  reduced_update_tuple_stm[6]);
        // reductionTree<7> (7, update_tuple_stm_array[7],  reduced_update_tuple_stm[7]);

        for(int i = 0; i < GATHER_PE_NUM; i++)
        {
        #pragma HLS UNROLL
            accGather(update_tuple_stm_l3[i], tmp_prop_stm[i], part_dst_offset);
        }
        writeResults(tmp_prop_stm, part_dst_offset, b_tmp_prop_stm);
    }
} // extern C