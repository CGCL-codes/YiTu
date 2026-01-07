#ifndef __ACC_SCATTER_H__
#define __ACC_SCATTER_H__
#include <ap_int.h>
#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "fpga_application.h"

template <typename T1, typename T2>
void stream2axistream(
                    hls::stream<T1> &stream,
                    hls::stream<T2> &axi_stream
                    ){
    
    stream2axistream:
    while (true){

        T1 tmp_t1 = stream.read();
        
        T2 tmp_t2;
        tmp_t2.data = tmp_t1.request_round;
        tmp_t2.last = tmp_t1.end_flag;
        write_to_stream(axi_stream, tmp_t2);
        if(tmp_t1.end_flag) break;
    }
}

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
        tmp_t2.addr = tmp_t1.dest;
        tmp_t2.end_flag = tmp_t1.last;
        write_to_stream(stream, tmp_t2);
        if(tmp_t2.end_flag) break;
    }
}

//Ping-Pong缓冲设计消除内存访问停顿 + 8个Scatter PE独立缓冲
void accScatter(
    hls::stream<ppb_request_dt>     &ppb_request_stm,
    hls::stream<ppb_response_dt>    &ppb_response_stm,
    hls::stream<edge_burst_dt>      &edge_burst_stm,
    hls::stream<update_set_dt>      &update_set_stm,
    ap_uint<32>                     part_edge_num
    )
{
    // as we can buffer two vertices in one row with width of 64-bit, we can let the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2.
    ap_uint<512> src_prop_buffer[SCATTER_PE_NUM][2][SRC_BUFFER_SIZE >> 4];
#pragma HLS ARRAY_PARTITION variable=src_prop_buffer dim=1 complete
#pragma HLS BIND_STORAGE variable=src_prop_buffer type = RAM_S2P impl = BRAM
#pragma HLS dependence variable=src_prop_buffer inter false    

    ap_uint<32> pp_read_idx = 0;
    ap_uint<32> pp_write_idx = 0;

    ap_uint<32> pp_reponse_idx = 0;

    ap_uint<32> pp_read_round = 0;
    ap_uint<32> pp_write_round = 0;

    ap_uint<32> pp_request_round = 0;

    ap_uint<32> edge_set_cnt = 0;
    
    bool wait_flag = 0;

    edge_burst_dt an_edge_burst;
    
    ppb_request_dt one_ppb_request;

    ppb_response_dt one_ppb_response;
    

    scatterLoop:
    while (true)
    {
#pragma HLS PIPELINE II=1
        // logic to fill the ping-pong buffer.
        //提前请求下一批顶点属性
        if((pp_request_round - pp_read_round) <= 1){
            if(pp_request_round < pp_read_round) pp_request_round = pp_read_round;
            one_ppb_request.request_round = pp_request_round;
            one_ppb_request.end_flag = 0;
            write_to_stream(ppb_request_stm, one_ppb_request);
            pp_request_round ++;
        }
        //将响应数据写入Ping-Pong缓冲
        if(read_from_stream_nb(ppb_response_stm, one_ppb_response)){
            
            pp_write_round = one_ppb_response.addr << 4 >> LOG2_SRC_BUFFER_SIZE;

            bool write_buffer = pp_write_round & 0x1;

            uint write_idx = one_ppb_response.addr & ((SRC_BUFFER_SIZE >> 4) - 1);

            ap_uint<512> one_read_burst = one_ppb_response.data; //src_prop[(base_addr >> 4) + pp_write_idx];

            for (int u = 0; u < SCATTER_PE_NUM; u++){
            #pragma HLS UNROLL
                src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;
            }
        }
        
        // logic to read the ping-pong buffer and synchronization.

        if(!wait_flag) read_from_stream(edge_burst_stm, an_edge_burst);

        pp_read_round = (an_edge_burst.edges[0].src.range(30, 0) / SRC_BUFFER_SIZE);

        if(pp_read_round >= pp_write_round) 
            wait_flag = 1; 
        else 
            wait_flag = 0;

        //DEBUG_PRINTF(" scatter pp_read_round = %d \n", (uint)pp_read_round);

        if(!wait_flag){

            bool read_buffer = pp_read_round & 0x1;
            
            update_set_dt an_update_set;
            //从缓冲读取源顶点属性，调用用户定义的PROP_COMPUTE_STAGE0生成更新值
            for (int u = 0; u < SCATTER_PE_NUM; u ++){ // "u" indicates the default unroll factor of the loop
            #pragma HLS UNROLL
                ap_uint<31> idx = (an_edge_burst.edges[u].src.range(30, 0) % SRC_BUFFER_SIZE);
                ap_uint<30> uram_row_idx = idx >> 4;
                ap_uint<30> uram_row_offset = (idx & 0xf);

                ap_uint<512> uram_row = src_prop_buffer[u][read_buffer][uram_row_idx];
                ap_uint<32> src_prop = uram_row.range(31 + (uram_row_offset << 5), (uram_row_offset << 5));  

                ap_uint<32> update = PROP_COMPUTE_STAGE0(src_prop);

                // ap_uint<32> update = uram_row.range(31 + (uram_row_offset << 5), (uram_row_offset << 5));
                
                an_update_set.tuples[u].update = update;
                an_update_set.tuples[u].dst = an_edge_burst.edges[u].dst;
            }
            write_to_stream(update_set_stm, an_update_set);
            edge_set_cnt ++;
        }
        //DEBUG_PRINTF("current processed sets of edges = %d \n", cnt++);

        if(edge_set_cnt >= (part_edge_num >> LOG2_NUM_EDGE_PER_BURST)){
            one_ppb_request.end_flag = 1;
            write_to_stream(ppb_request_stm, one_ppb_request);
            exitscatter:
            while(1){ 
                read_from_stream(ppb_response_stm, one_ppb_response);
                if (one_ppb_response.end_flag) break;    
            }
            DEBUG_PRINTF("function accScatter exits!\n");
            break;
        }  
    }
}

#endif /* __ACC_SCATTER_H__ */