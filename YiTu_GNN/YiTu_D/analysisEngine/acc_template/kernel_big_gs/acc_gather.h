#ifndef __ACC_GATHER_H__
#define __ACC_GATHER_H__

#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "l2.h"
#include "fpga_application.h"
#include "acc_data_types.h"

// The input to gather is an update tuple with format of <update, destination vertex ID (offset maybe)>
// The gather will output all accumulated value to the HBM or down-stream. 
//accGather(part_dst_offset, update_set_stm, dst_tmp_prop);

//实现2x2交换单元
//omega network 
void sender(
    int i,
    hls::stream<update_tuple_dt>    &update_set_stm_in1,
    hls::stream<update_tuple_dt>    &update_set_stm_in2,
    hls::stream<update_tuple_dt>    &update_set_stm_out1,
    hls::stream<update_tuple_dt>    &update_set_stm_out2,
    hls::stream<update_tuple_dt>    &update_set_stm_out3,
    hls::stream<update_tuple_dt>    &update_set_stm_out4
    ){
#pragma HLS function_instantiate variable=i
    bool in1_end_flag = 0;
    bool in2_end_flag = 0;

    sender:
    while(true){
    #pragma HLS PIPELINE II=1
        update_tuple_dt data1, data2;  

        if(!update_set_stm_in1.empty()){
            read_from_stream(update_set_stm_in1, data1);
            
            if(!data1.end_flag){

                if((data1.dst >> i) & 0x1)
                    write_to_stream(update_set_stm_out2, data1);
                else 
                    write_to_stream(update_set_stm_out1, data1);
            }
            else {
                in1_end_flag = data1.end_flag;
            }
        }

        if(!update_set_stm_in2.empty()){
            read_from_stream(update_set_stm_in2, data2);
            if(!data2.end_flag){
                if((data2.dst >> i) & 0x1)
                    write_to_stream(update_set_stm_out4, data2);
                else 
                    write_to_stream(update_set_stm_out3, data2);
            }
            else {
                in2_end_flag = data2.end_flag;
            }
        }

        if(in1_end_flag && in2_end_flag){
            update_tuple_dt data;
            data.end_flag = 1;
            write_to_stream(update_set_stm_out4, data);
            write_to_stream(update_set_stm_out1, data);
            write_to_stream(update_set_stm_out2, data);
            write_to_stream(update_set_stm_out3, data);
            //DEBUG_PRINTF("mergeStream2x1 module exits!\n");
            in1_end_flag = 0;
            in2_end_flag = 0;
            break;
        }
    }        
}


void receiver(
    int i,
    hls::stream<update_tuple_dt>    &update_set_stm_out1,
    hls::stream<update_tuple_dt>    &update_set_stm_out2,
    hls::stream<update_tuple_dt>    &update_set_stm_in1,
    hls::stream<update_tuple_dt>    &update_set_stm_in2,
    hls::stream<update_tuple_dt>    &update_set_stm_in3,
    hls::stream<update_tuple_dt>    &update_set_stm_in4
    ){
#pragma HLS function_instantiate variable=i
    bool in1_end_flag = 0;
    bool in2_end_flag = 0;
    bool in3_end_flag = 0;
    bool in4_end_flag = 0;

    while(true){
    #pragma HLS PIPELINE II=1
        update_tuple_dt data1, data2;  

        if(!update_set_stm_in1.empty()){
            update_tuple_dt data;
            read_from_stream(update_set_stm_in1, data);
            if(!data.end_flag)             
                write_to_stream(update_set_stm_out1, data);
            else in1_end_flag = data.end_flag;
        }
        else if(!update_set_stm_in3.empty()){
            update_tuple_dt data;
            read_from_stream(update_set_stm_in3, data);
            if(!data.end_flag)             
                write_to_stream(update_set_stm_out1, data);
            else in3_end_flag = data.end_flag;        
        }

        if(!update_set_stm_in2.empty()){
            update_tuple_dt data;
            read_from_stream(update_set_stm_in2, data);
            if(!data.end_flag)             
                write_to_stream(update_set_stm_out2, data);
            else in2_end_flag = data.end_flag;
        }
        else if(!update_set_stm_in4.empty()){
            update_tuple_dt data;
            read_from_stream(update_set_stm_in4, data);
            if(!data.end_flag)             
                write_to_stream(update_set_stm_out2, data);
            else in4_end_flag = data.end_flag;
        }

        if(in1_end_flag && in2_end_flag && in3_end_flag && in4_end_flag){
            update_tuple_dt data;
            data.end_flag = 1;
            write_to_stream(update_set_stm_out1, data);
            write_to_stream(update_set_stm_out2, data);
            break;
        }
    }        
}

//构建完整的交换网络
template <int an_unused_template_parameter>
void switch2x2(
    int i,
    hls::stream<update_tuple_dt>    &update_set_stm_in1,
    hls::stream<update_tuple_dt>    &update_set_stm_in2,
    hls::stream<update_tuple_dt>    &update_set_stm_out1,
    hls::stream<update_tuple_dt>    &update_set_stm_out2
    ){

//#pragma HLS function_instantiate variable=i
        
        hls::stream<update_tuple_dt> l1_1;
#pragma HLS stream variable=l1_1 depth=2
        hls::stream<update_tuple_dt> l1_2;
#pragma HLS stream variable=l1_2 depth=2
        hls::stream<update_tuple_dt> l1_3;
#pragma HLS stream variable=l1_3 depth=2
        hls::stream<update_tuple_dt> l1_4;
#pragma HLS stream variable=l1_4 depth=2

// hls::stream<update_tuple_dt>       l2_update_tuples[2];
// #pragma HLS stream variable=l2_update_tuples depth=2
#pragma HLS DATAFLOW
        sender(i, update_set_stm_in1, update_set_stm_in2, l1_1, l1_2, l1_3, l1_4);
        receiver(i, update_set_stm_out1, update_set_stm_out2, l1_1, l1_2, l1_3, l1_4);
}

//将更新集合分发到多个流
void dispatchUpdateTuples(
    hls::stream<update_set_dt>    &update_set_stm,
    hls::stream<update_tuple_dt>  (&update_tuple_stm)[NUM_EDGE_PER_BURST],
    ap_uint<32>                   part_edge_num
){

    dispatchUpdateTuples:
    for(int i = 0; i < (part_edge_num >> LOG2_NUM_EDGE_PER_BURST); i ++) // 8 edges per read, hence << 3, it needs to be tuned according to the data width of edges.
    {
#pragma HLS PIPELINE II=1

        update_set_dt an_update_set;

        read_from_stream(update_set_stm, an_update_set);

        for (int u = 0; u < NUM_EDGE_PER_BURST; u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL 
            if (IS_ACTIVE_VERTEX(an_update_set.tuples[u].update))
                write_to_stream(update_tuple_stm[u], an_update_set.tuples[u]);
        }   
    }

    {    
        for (int u = 0; u < NUM_EDGE_PER_BURST; u ++){ // "u" indicates the default unroll factor of the loop
        #pragma HLS UNROLL 
            update_tuple_dt an_update_tuple;
            an_update_tuple.end_flag = 1;
            write_to_stream(update_tuple_stm[u], an_update_tuple);  
        }              
    }
} 

//合并两个流为一个
void mergeStream2x1(
    int i,
    hls::stream<update_tuple_dt> &in1,
    hls::stream<update_tuple_dt> &in2,
    hls::stream<update_tuple_dt> &out
                ){
#pragma HLS function_instantiate variable=i
    bool in1_end_flag = 0;
    bool in2_end_flag = 0;
    //uint cnt = 0;
    mergeStream2x1:
    while(true){
    #pragma HLS PIPELINE II=2
        update_tuple_dt data1, data2;  
        bool in1_process_flag = read_from_stream_nb(in1, data1);
        bool in2_process_flag = read_from_stream_nb(in2, data2);
        // in1_process_flag      
        // if(read_from_stream_nb(in1, data1) == 0) write_to_stream(out, data1);
        // if(read_from_stream_nb(in2, data2) == 0) write_to_stream(out, data2);
        if(in1_process_flag && data1.end_flag) in1_end_flag = 1;
        if(in2_process_flag && data2.end_flag) in2_end_flag = 1;

        if(in1_process_flag && (!in1_end_flag)) write_to_stream(out, data1);
        if(in2_process_flag && (!in2_end_flag)) write_to_stream(out, data2);
        //DEBUG_PRINTF("current processed dst: %d! \n", (uint)data1.dst);

        if(in1_end_flag && in2_end_flag){
            update_tuple_dt data;
            data.end_flag = 1;
            write_to_stream(out, data);
            //DEBUG_PRINTF("mergeStream2x1 module exits!\n");
            in1_end_flag = 0;
            in2_end_flag = 0;
            break;
        }
        //cnt ++;
    }
}

//构建多级归约树
template <int an_unused_template_parameter>
void reductionTree(
    int i,
    hls::stream<update_tuple_dt>    (&update_set_stm)[NUM_EDGE_PER_BURST],
    hls::stream<update_tuple_dt>    &reduced_update_tuple_stm
    ){
        hls::stream<update_tuple_dt>       l1_update_tuples[4];
#pragma HLS stream variable=l1_update_tuples depth=2

        hls::stream<update_tuple_dt>       l2_update_tuples[2];
#pragma HLS stream variable=l2_update_tuples depth=2

#pragma HLS DATAFLOW

        //mergeStream8x4(i, update_set_stm, l1);
        //mergeStream4x2(l1, l2);
        mergeStream2x1(7*i+0, update_set_stm[0], update_set_stm[4], l1_update_tuples[0]);
        mergeStream2x1(7*i+1, update_set_stm[2], update_set_stm[6], l1_update_tuples[1]);
        mergeStream2x1(7*i+2, update_set_stm[1], update_set_stm[5], l1_update_tuples[2]);
        mergeStream2x1(7*i+3, update_set_stm[3], update_set_stm[7], l1_update_tuples[3]);

        mergeStream2x1(7*i+4, l1_update_tuples[0], l1_update_tuples[1], l2_update_tuples[0]);
        mergeStream2x1(7*i+5, l1_update_tuples[2], l1_update_tuples[3], l2_update_tuples[1]);
        
        mergeStream2x1(7*i+6, l2_update_tuples[0], l2_update_tuples[1], reduced_update_tuple_stm);
}

//累积更新到目标顶点属性
void accGather(
    //ap_uint<64> dst_tmp_prop_buffer[(BIG_KERNEL_DST_BUFFER_SIZE >> 1) >> LOG2_GATHER_PE_NUM],    
    hls::stream<update_tuple_dt> &update_tuple_stm,
    hls::stream<ap_uint<64>> &tmp_prop_stm,
    uint32_t                 part_dst_offset
){
//#pragma HLS function_instantiate variable=update_tuple_stm
// as we can buffer two vertices in one row with width of 64-bit, we can let the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2.
    ap_uint<64> dst_tmp_prop_buffer[(BIG_KERNEL_DST_BUFFER_SIZE >> 1) >> LOG2_GATHER_PE_NUM]; 
#pragma HLS BIND_STORAGE variable = dst_tmp_prop_buffer type = RAM_2P impl = URAM
#pragma HLS dependence variable=dst_tmp_prop_buffer inter false

#ifdef SW_EMU
    for (int i = 0; i < ((BIG_KERNEL_DST_BUFFER_SIZE >> 1) >> LOG2_GATHER_PE_NUM); i ++){
    #pragma HLS UNROLL factor=1
            dst_tmp_prop_buffer[i].range(31,0) = 0;
            dst_tmp_prop_buffer[i].range(63,32) = 0;
    }
#endif

    ap_uint<64> last_value[L + 1]; // we do not init it at this version. 
    uint last_index[L + 1];     // we do not init it at this version. 
#pragma HLS ARRAY_PARTITION variable=last_value dim=0 complete
#pragma HLS ARRAY_PARTITION variable=last_index dim=0 complete


    for (int i = 0; i < (L + 1); i ++){
    #pragma HLS UNROLL
            last_value[i] = 0;
            last_index[i] = 0;
    }

int cnt = 0;
    while (true) {
#pragma HLS PIPELINE II=1
        
        update_tuple_dt one_update_tuple;
        read_from_stream(update_tuple_stm, one_update_tuple);
        
        //DEBUG_PRINTF("processed TUPLES: %d! flag %d \n",  (uint)cnt, one_update_tuple.end_flag);

        if (one_update_tuple.end_flag) {
            //DEBUG_PRINTF("gather exits! \n");
            //DEBUG_PRINTF("processed TUPLES: %d! \n", (uint)cnt);
            break;
        }
        cnt ++;

        if (!(one_update_tuple.dst.range(19, 19) & 0x1)) // if they are not dummy tuple, we will process it
        {   
            ap_uint<32> dst_idx  = ((one_update_tuple.dst) >> LOG2_GATHER_PE_NUM); 
            ap_uint<32> uram_addr = (dst_idx >> 1) & (((BIG_KERNEL_DST_BUFFER_SIZE) - 1) >> LOG2_GATHER_PE_NUM);
            ap_uint<32> tuple_update_value = one_update_tuple.update;

            ap_uint<64> updated_value = dst_tmp_prop_buffer[uram_addr];
            
            for (int i = 0; i < L + 1; i ++)
        #pragma HLS UNROLL
            {
                if (last_index[i] == uram_addr) updated_value = last_value[i];
            }
            
            for (int i = 0; i < L; i ++)
        #pragma HLS UNROLL
            {
                last_value[i] = last_value[i + 1];
                last_index[i] = last_index[i + 1];
            }

            ap_uint<64> temp_updated_value = updated_value;

            uint msb = updated_value.range(63, 32);
            uint lsb = updated_value.range(31, 0);

            uint msb_out = PROP_COMPUTE_STAGE3(msb, tuple_update_value);
            uint lsb_out = PROP_COMPUTE_STAGE3(lsb, tuple_update_value);
            // uint msb_out = (msb + tuple_update_value);
            // uint lsb_out = (lsb + tuple_update_value);

            ap_uint<64> accumulate_msb;
            ap_uint<64> accumulate_lsb;

            accumulate_msb.range(63, 32) = msb_out;
            accumulate_msb.range(31,  0) = temp_updated_value.range(31, 0);

            accumulate_lsb.range(63, 32) = temp_updated_value.range(63, 32);
            accumulate_lsb.range(31,  0) = lsb_out;

            if (dst_idx & 0x01)
            {
                dst_tmp_prop_buffer[uram_addr] = accumulate_msb;
                last_value[L] = accumulate_msb;
            }
            else
            {
                dst_tmp_prop_buffer[uram_addr] = accumulate_lsb;
                last_value[L] = accumulate_lsb;
            }
            last_index[L] = uram_addr;
        }
    }

    // write results back after the gather stage
    // merge can be costly than shuffling based solution since we cannot read 512 bits per cycle from URAM. 
    tmpPropWriteToStream:
    for (int i = 0; i < ((BIG_KERNEL_DST_BUFFER_SIZE >> 1) >> LOG2_GATHER_PE_NUM); i++)  //  ap_uint<512> / ap_uint<64>
    {
#pragma HLS UNROLL factor=1
        ap_uint<64> tmp_uram_data = dst_tmp_prop_buffer[i];
        dst_tmp_prop_buffer[i] = 0;
        write_to_stream(tmp_prop_stm, tmp_uram_data);
    }
}

//将结果写入输出流
void writeResults(
    hls::stream<ap_uint<64>> (&tmp_prop_stm)[GATHER_PE_NUM],
    uint32_t                    part_dst_offset,
    hls::stream<b_tmp_prop_pkt>       &big_pipe_tmp_prop_stm
){

    tmpPropWriteBack:
    for (int i = 0; i < ((BIG_KERNEL_DST_BUFFER_SIZE >> 1) >> LOG2_GATHER_PE_NUM); i++)  //  ap_uint<512> / ap_uint<64>
    {
#pragma HLS PIPELINE
        b_tmp_prop_pkt one_write_burst;
        ap_uint<64> tmp_uram_data[GATHER_PE_NUM];

        tmpPropWriteBackInner:
        for (int u = 0; u < GATHER_PE_NUM; u++){
    #pragma HLS UNROLL
            //ap_uint<64> tmp_uram_data;
            read_from_stream(tmp_prop_stm[u], tmp_uram_data[u]);
            //one_write_burst.range(63 + (u << 6),  (u << 6)) = tmp_uram_data;
            one_write_burst.data.range(31 + (u << 5),  (u << 5)) = tmp_uram_data[u].range(31, 0);
            one_write_burst.data.range(31 + (u << 5) + 256,  (u << 5) + 256) = tmp_uram_data[u].range(63, 32);
        }
        
        write_to_stream(big_pipe_tmp_prop_stm, one_write_burst);
    }

    DEBUG_PRINTF("big kernel writeResults module exits!\n");
}
#endif  // __ACC_GATHER_H__