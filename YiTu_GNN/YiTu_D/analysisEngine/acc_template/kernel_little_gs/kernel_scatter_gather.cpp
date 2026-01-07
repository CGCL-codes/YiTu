#include <hls_stream.h>
#include <string.h>

#include "acc_data_types.h"

#include "l1_api.h"
#include "acc_config.h"

#include "acc_scatter.h"
#include "acc_gather.h"

/*
1.边数据读取
2.流格式转换
3.Scatter
4.Gather
5.合并结果写入
*/
extern "C" {
    void  littleKernelScatterGather(
        edge_burst_dt   *part_edge_array,
        uint             part_edge_num,
        uint             part_dst_offset,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm,
        hls::stream<l_tmp_prop_pkt> &l_tmp_prop_stm
        // the offset of destination vertices buffered in current gs kernel. e.g., 0 to 0 + BUFFERED_MAX_NUM_VERTEX
    )
    {
        const int stream_depth = 8;
        const int large_stream_depth = 8;

#pragma HLS DATAFLOW

#pragma HLS INTERFACE m_axi port=part_edge_array offset=slave bundle=gmem0
#pragma HLS INTERFACE s_axilite port=part_edge_array bundle=control

#pragma HLS INTERFACE s_axilite port=part_edge_num        bundle=control
#pragma HLS INTERFACE s_axilite port=part_dst_offset      bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

        hls::stream<edge_burst_dt>           edge_burst_stm;
#pragma HLS stream variable=edge_burst_stm  depth=large_stream_depth 


        hls::stream<ppb_request_dt>           ppb_request_stm;
#pragma HLS stream variable=ppb_request_stm  depth=large_stream_depth 

        hls::stream<ppb_response_dt>           ppb_response_stm;
#pragma HLS stream variable=ppb_response_stm  depth=large_stream_depth 


        hls::stream<update_set_dt>           update_set_stm;
#pragma HLS stream variable=update_set_stm depth=stream_depth

        hls::stream<ap_uint<64>>       tmp_prop_stm[GATHER_PE_NUM];
#pragma HLS stream variable=tmp_prop_stm depth=stream_depth

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
        }
        DEBUG_PRINTF("reading edge exits! with %d edge sets read.\n", (part_edge_num >> LOG2_NUM_EDGE_PER_BURST));

        stream2axistream(ppb_request_stm, l_ppb_request_stm);
        
        axistream2stream(l_ppb_response_stm, ppb_response_stm);

        accScatter(ppb_request_stm, ppb_response_stm, edge_burst_stm, update_set_stm, part_edge_num);

        accGather(part_edge_num, update_set_stm, tmp_prop_stm);

        mergeWriteResults(tmp_prop_stm, part_dst_offset, l_tmp_prop_stm);
    }
} // extern C