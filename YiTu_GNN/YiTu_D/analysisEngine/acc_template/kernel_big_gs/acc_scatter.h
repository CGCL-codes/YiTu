#ifndef __ACC_SCATTER_H__
#define __ACC_SCATTER_H__

#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "fpga_application.h"

// this is function is special to our case!!!
//辅助函数，计算前导零用于请求优化
uchar countLeadingZeros(ap_uint<NUM_EDGE_PER_BURST> bit_mask){
#pragma HLS INLINE
    uchar cnt_leading_zeros = 0;
    switch (bit_mask)
    {
    case 0:  // not possible since there is no all zeros set sent to memory request kernel.
        cnt_leading_zeros = 8;
        break;
    case 1:
        cnt_leading_zeros = 7;
        break;
    case 3:
        cnt_leading_zeros = 6;
        break;
    case 7:
        cnt_leading_zeros = 5;
        break;
    case 15:
        cnt_leading_zeros = 4;
        break;
    case 31:
        cnt_leading_zeros = 3;
        break;
    case 63:
        cnt_leading_zeros = 2;
        break;
    case 127:
        cnt_leading_zeros = 1;
        break;
    case 255:
        cnt_leading_zeros = 0;
        break;
    default:
        break;
    }
    return cnt_leading_zeros;
}

//普通流转化为AXI流
template <typename T1, typename T2>
void stream2axistream(
                    hls::stream<T1> &stream,
                    hls::stream<T2> &axi_stream
                    ){
    
    stream2axistream:
    while (true){

        T1 tmp_t1 = stream.read();
        
        T2 tmp_t2;
        tmp_t2.data = tmp_t1.idx;
        tmp_t2.dest = tmp_t1.dst;
        tmp_t2.last = tmp_t1.end_flag;
        
        write_to_stream(axi_stream, tmp_t2);

        if(tmp_t1.end_flag) break;
    }
}


//AXI流转化为普通流
template <typename T1, typename T2>
void axistream2stream(
                    hls::stream<T1> &axi_stream,
                    hls::stream<T2> &stream
                    ){
    
    axistream2stream:
    while (true){

        T1 tmp_t1 = axi_stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.data;
        tmp_t2.dst = tmp_t1.dest;
        tmp_t2.end_flag = tmp_t1.last;
        
        write_to_stream(stream, tmp_t2);
        if(tmp_t2.end_flag) break;
    }
}


//根据源顶点ID生成内存请求
void genMemRequest(
    hls::stream<src_burst_dt>       &src_burst_stm,
    hls::stream<mem_request_burst_dt>  &mem_request_burst_stm,
    ap_uint<32>                     part_edge_num
    )
{   

    // ap_uint<26>  last_cacheline_idx_min = 0;
    ap_uint<26>  last_cacheline_idx_max = 0;

    genMemRequest: 
    for(int i = 0; i < (part_edge_num >> LOG2_NUM_EDGE_PER_BURST); i ++) {
        //接收源顶点ID流
        src_burst_dt an_src_burst; //input set
        read_from_stream(src_burst_stm, an_src_burst);
        
        // calculate the cacheline indices of srcs ...
        ap_uint<26> cacheline_idx[NUM_EDGE_PER_BURST]; // we calculate the offsets of all requests compared with last max.
    #pragma HLS ARRAY_PARTITION variable=cacheline_idx dim=0 complete
        for (int u = 0; u < NUM_EDGE_PER_BURST; u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL
            cacheline_idx[u] = an_src_burst.src[u].range(30, 0) >> 4;
        }        
        
        // calculate the offsets of cacheline indices of srcs to the max cacheline index of the last src burst...
        ap_uint<26> cacheline_idx_offset[NUM_EDGE_PER_BURST]; // we calculate the offsets of all requests compared with last max.
    #pragma HLS ARRAY_PARTITION variable=cacheline_idx_offset dim=0 complete
        for (int u = 0; u < NUM_EDGE_PER_BURST; u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL
            cacheline_idx_offset[u] = cacheline_idx[u] - last_cacheline_idx_max;
        }
        
        // send out the requests if not all of the offets are zeros...
        if (cacheline_idx_offset[NUM_EDGE_PER_BURST - 1]){ // if the max cacheline index equals to the previous one, then we skip it. otherwise, we send memory requests. 
            // prepare a set of requests for current src set.
            ap_uint<NUM_EDGE_PER_BURST> bit_mask;
            for (int u = 0; u < NUM_EDGE_PER_BURST; u ++) { // "u" indicates the default unroll factor of the loop
            #pragma HLS UNROLL
                if(cacheline_idx_offset[u] == 0) 
                    bit_mask.range(u,u) = 1;
                else 
                    bit_mask.range(u,u) = 0;
            }
            //确定需要发送的请求数量
            ap_uint<4> num_read = countLeadingZeros(bit_mask);

            mem_request_burst_dt an_mem_request_burst;
            an_mem_request_burst.offset = NUM_EDGE_PER_BURST - num_read;
            an_mem_request_burst.end_flag = 0;
                 
            for (int u = 0; u < NUM_EDGE_PER_BURST; u ++) { // "u" indicates the default unroll factor of the loop
            #pragma HLS UNROLL
                an_mem_request_burst.idx[u] = cacheline_idx[u];
            }
            //输出内存请求流
            write_to_stream(mem_request_burst_stm, an_mem_request_burst);
            //if (an_src_burst.src[NUM_EDGE_PER_BURST-1].range(31, 31) & 0x1) 
                //DEBUG_PRINTF("write flag to readMem module! with dst = 0x%x \n", (uint)an_src_burst.src[NUM_EDGE_PER_BURST-1]);
        }

        last_cacheline_idx_max = cacheline_idx[NUM_EDGE_PER_BURST - 1];
        
        // exit condition: the last edge has destination vertex whose most significant bit is one.
    }

    {
        mem_request_burst_dt an_mem_request_burst;
        an_mem_request_burst.offset = NUM_EDGE_PER_BURST - 1;
        an_mem_request_burst.end_flag = 1;
        write_to_stream(mem_request_burst_stm, an_mem_request_burst);
        //DEBUG_PRINTF("genMemRequest module exits! \n");
    }
}

//发送缓存线请求到内存系统
void sendCachelineRequest(
    hls::stream<mem_request_burst_dt>   &mem_request_burst_stm,
    hls::stream<cacheline_dt>   &cacheline_stm
    //hls::stream<big_krnl_src_prop_request_pkt>   &big_krnl_src_prop_request_stm
    )
{  
    bool end_flag = 0;


    ap_uint<26> cacheline_idx[NUM_EDGE_PER_BURST];
#pragma HLS ARRAY_PARTITION variable=cacheline_idx dim=0 complete

    cacheline_dt a_cacheline;
    a_cacheline.end_flag = 0;
    a_cacheline.idx = 0;
    write_to_stream(cacheline_stm, a_cacheline);

    sendCachelineRequestOuter: 
    while(true){
#pragma HLS PIPELINE II=1
#pragma HLS dependence variable=cacheline_idx inter false

        mem_request_burst_dt an_mem_request_burst;        
        read_from_stream(mem_request_burst_stm, an_mem_request_burst);

        //ap_uint<3> start_pe_id =  an_mem_request_burst.offset;

        for (int u = 0; u < NUM_EDGE_PER_BURST; u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL
            cacheline_idx[u] = an_mem_request_burst.idx[u];
        }
        
        {
            sendCachelineRequestInner:
            for (ap_uint<4> i = an_mem_request_burst.offset; i < NUM_EDGE_PER_BURST; i ++){
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS unroll factor=1
                a_cacheline.idx = cacheline_idx[i]; 
                a_cacheline.dst = i;
                a_cacheline.end_flag = an_mem_request_burst.end_flag;
                write_to_stream(cacheline_stm, a_cacheline);
            }
        }

        if(an_mem_request_burst.end_flag){
            //DEBUG_PRINTF("readMem module exits!\n");
            break;
        }
    }
}

//接收内存响应并分发到各处理单元
void receiveResponses(
                hls::stream<cacheline_response_dt>   &cacheline_response_stm,
                hls::stream<ap_uint<512>> (&spe_cacheline_stm)[SCATTER_PE_NUM]
                ){ 
                    
    cacheline_response_dt one_cacheline_response;
    read_from_stream(cacheline_response_stm, one_cacheline_response);

    ap_uint<512> first_cacheline = one_cacheline_response.data;
    for (int u = 0; u < SCATTER_PE_NUM; u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL  
        write_to_stream(spe_cacheline_stm[u], first_cacheline);
    }

    // ap_uint<26> last_requested_cacheline_idx = 0;
    // ap_uint<512> last_requested_cacheline;
    receiveResponses:
    while(true){
#pragma HLS PIPELINE II=1
        if(read_from_stream_nb(cacheline_response_stm, one_cacheline_response))
        {
            if(one_cacheline_response.end_flag) break;
            
            ap_uint<512> cacheline = one_cacheline_response.data;
            ap_uint<8> dst = one_cacheline_response.dst;
            write_to_stream(spe_cacheline_stm[dst], cacheline);
        }
    }
}

//处理边数据，生成更新元组
void  accScatter(
    hls::stream<ap_uint<512>>       (&spe_cacheline_stm)[SCATTER_PE_NUM],
    hls::stream<edge_burst_dt>      &edge_burst_stm,
    hls::stream<update_set_dt>      &update_set_stm,
    ap_uint<32>                     part_edge_num
    )
{   

    ap_uint<512>  last_cacheline[SCATTER_PE_NUM] = {0};
    #pragma HLS ARRAY_PARTITION variable=last_cacheline dim=0 complete
        uint last_cacheline_idx[SCATTER_PE_NUM] = {0};
    #pragma HLS ARRAY_PARTITION variable=last_cacheline_idx dim=0 complete

//#ifdef SW_EMU
    // init the first cacheline
    for (int j = 0; j < SCATTER_PE_NUM; j ++) {
    #pragma HLS UNROLL
        read_from_stream(spe_cacheline_stm[j], last_cacheline[j]);
        last_cacheline_idx[j] = 0x0;
    }

#ifdef SW_EMU
    int cnt = 0;
#endif
    scatterLoop: 
    for(int i = 0; i < (part_edge_num >> LOG2_NUM_EDGE_PER_BURST); i ++) {
        
        //src_prop_burst_dt an_src_prop_burst;
        edge_burst_dt an_edge_burst;

        update_set_dt an_update_set;

        //read_from_stream(src_prop_stm, an_src_prop_burst);
        read_from_stream(edge_burst_stm, an_edge_burst);

        // store the latest cacheline for every Scatter PE

        for (int u = 0; u < SCATTER_PE_NUM; u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL

            uint cacheline_idx = an_edge_burst.edges[u].src.range(30, 0) >> 4;
            uint cacheline_offset = an_edge_burst.edges[u].src.range(30, 0) & 0xf;

            ap_uint<512> cacheline;

            if(cacheline_idx == last_cacheline_idx[u]){
                cacheline = last_cacheline[u];
            } else {
                read_from_stream(spe_cacheline_stm[u], cacheline);
            }

            ap_uint<32> src_prop = cacheline.range(31 + (cacheline_offset << 5), (cacheline_offset << 5));
            ap_uint<32> update = PROP_COMPUTE_STAGE0(src_prop);

            //ap_uint<32> update = cacheline.range(31 + (cacheline_offset << 5), (cacheline_offset << 5));
            
            an_update_set.tuples[u].end_flag = 0;
            an_update_set.tuples[u].update = update;
            an_update_set.tuples[u].dst = an_edge_burst.edges[u].dst;

            if (u == SCATTER_PE_NUM-1) {
                last_cacheline[u] = cacheline;
                last_cacheline_idx[u] = cacheline_idx;
            } 
        }
        write_to_stream(update_set_stm, an_update_set);

        //update all cachelines to be maximal one
        for (int u = 0; u < (SCATTER_PE_NUM-1); u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL
            last_cacheline[u] = last_cacheline[SCATTER_PE_NUM-1];
            last_cacheline_idx[u] = last_cacheline_idx[SCATTER_PE_NUM-1];            
        }
        // DEBUG_PRINTF("current scatter burst num =  %d \n", cnt ++);
        // exit condition: the last edge has destination vertex whose most significant bit is one.
        //DEBUG_PRINTF("current processed sets of edges = %d \n", cnt++);
    }
}
 
#endif /* __ACC_SCATTER_H__ */