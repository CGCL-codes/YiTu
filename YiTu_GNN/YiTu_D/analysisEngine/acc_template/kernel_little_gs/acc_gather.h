#ifndef __ACC_GATHER_H__
#define __ACC_GATHER_H__

#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "fpga_application.h"
#include "acc_data_types.h"

// The input to gather is an update tuple with format of <update, destination vertex ID (offset maybe)>
// The gather will output all accumulated value to the HBM or down-stream. 
//accGather(part_dst_offset, update_set_stm, dst_tmp_prop);

//Gather阶段
void accGather(
    //uint32_t                    part_dst_offset,
    ap_uint<32>                 part_edge_num,
    hls::stream<update_set_dt>  &update_set_stm,
    hls::stream<ap_uint<64>>    (&tmp_prop_stm)[GATHER_PE_NUM]
){

    // as we can buffer two vertices in one row with width of 64-bit, we can let the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2.
    ap_uint<64> dst_tmp_prop_buffer[GATHER_PE_NUM][(LITTLE_KERNEL_DST_BUFFER_SIZE >> 1)];
#pragma HLS ARRAY_PARTITION variable=dst_tmp_prop_buffer dim=1 complete
//#pragma HLS RESOURCE variable=dst_tmp_prop_buffer core=XPM_MEMORY uram   
#pragma HLS BIND_STORAGE variable = dst_tmp_prop_buffer type = RAM_S2P impl = URAM
#pragma HLS dependence variable=dst_tmp_prop_buffer inter false

#ifdef SW_EMU
    for (int i = 0; i < (LITTLE_KERNEL_DST_BUFFER_SIZE >> 1); i ++)
        for (int j = 0; j < GATHER_PE_NUM; j ++){
        #pragma HLS UNROLL
            dst_tmp_prop_buffer[j][i].range(31,0) = 0;
            dst_tmp_prop_buffer[j][i].range(63,32) = 0;
        }
#endif

    ap_uint<64> last_value[GATHER_PE_NUM][L + 1]; // we do not init it at this version. 
    uint last_index[GATHER_PE_NUM][L + 1];     // we do not init it at this version. 
#pragma HLS ARRAY_PARTITION variable=last_value dim=0 complete
#pragma HLS ARRAY_PARTITION variable=last_index dim=0 complete

//#ifdef SW_EMU
    for (int i = 0; i < (L + 1); i ++)
        for (int j = 0; j < GATHER_PE_NUM; j ++){
        #pragma HLS UNROLL
            last_value[j][i] = 0;
            last_index[j][i] = 0;
        }
//#endif
    gatherLoop:
    for(int i = 0; i < (part_edge_num >> LOG2_NUM_EDGE_PER_BURST); i ++) {
#pragma HLS PIPELINE II=1
        
        update_set_dt one_update_set;
        read_from_stream(update_set_stm, one_update_set);

        // exit condition: the last tuple has destination vertex whose most significant bit is one.
        for (int u = 0; u < GATHER_PE_NUM; u ++) { // "u" indicates the default unroll factor of the loop
    #pragma HLS UNROLL
            if ( (!(one_update_set.tuples[u].dst.range(19, 19) & 0x1)) && (IS_ACTIVE_VERTEX(one_update_set.tuples[u].update)) ) // if they are not dummy tuple, we will process it
            {   
                ap_uint<32> dst_idx  = one_update_set.tuples[u].dst & ((LITTLE_KERNEL_DST_BUFFER_SIZE) - 1);
                ap_uint<32> uram_addr = (dst_idx >> 1);
                ap_uint<32> tuple_update_value = one_update_set.tuples[u].update;

                ap_uint<64> updated_value = dst_tmp_prop_buffer[u][uram_addr];
                
                for (int i = 0; i < L + 1; i ++)
            #pragma HLS UNROLL
                {
                    if (last_index[u][i] == uram_addr) updated_value = last_value[u][i];
                }
                
                for (int i = 0; i < L; i ++)
            #pragma HLS UNROLL
                {
                    last_value[u][i] = last_value[u][i + 1];
                    last_index[u][i] = last_index[u][i + 1];
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
                    dst_tmp_prop_buffer[u][uram_addr] = accumulate_msb;
                    last_value[u][L] = accumulate_msb;
                }
                else
                {
                    dst_tmp_prop_buffer[u][uram_addr] = accumulate_lsb;
                    last_value[u][L] = accumulate_lsb;
                }
                last_index[u][L] = uram_addr;
            }
        }
    }

    // write results back after the gather stage
    // merge can be costly than shuffling based solution since we cannot read 512 bits per cycle from URAM. 
    tmpPropWriteToStream:
    for (int i = 0; i < ((LITTLE_KERNEL_DST_BUFFER_SIZE >> 1)); i++)  //  ap_uint<512> / ap_uint<64>
    {
#pragma HLS PIPELINE
        for (int u = 0; u < GATHER_PE_NUM; u++){
    #pragma HLS UNROLL
            ap_uint<64> tmp_uram_data = dst_tmp_prop_buffer[u][i];
            dst_tmp_prop_buffer[u][i] = 0;
            write_to_stream(tmp_prop_stm[u], tmp_uram_data);
        }
    }
}

//合并8个Gather PE的结果
void mergeWriteResults(
    hls::stream<ap_uint<64>> (&tmp_prop_stm)[GATHER_PE_NUM],
    uint32_t                    part_dst_offset,
    hls::stream<l_tmp_prop_pkt>    &little_pipe_tmp_prop_stm
    ){

    tmpPropWriteBack:
    for (int i = 0; i < ((LITTLE_KERNEL_DST_BUFFER_SIZE >> 1)); i++)  //  ap_uint<512> / ap_uint<64>
    {
#pragma HLS PIPELINE

        l_tmp_prop_pkt one_write_burst;

        uint uram_row_left = 0;
        uint uram_row_right = 0;
        for (int u = 0; u < GATHER_PE_NUM; u++){
    #pragma HLS UNROLL
            ap_uint<64> tmp_uram_data;
            read_from_stream(tmp_prop_stm[u], tmp_uram_data);
            // uram_row_right += tmp_uram_data.range(31, 0);       
            // uram_row_left += tmp_uram_data.range(63, 32);
            uram_row_right = PROP_COMPUTE_STAGE3(uram_row_right, tmp_uram_data.range(31, 0));
            uram_row_left = PROP_COMPUTE_STAGE3(uram_row_left, tmp_uram_data.range(63, 32));
        }
        ap_uint<64> merged_uram_data;
        merged_uram_data.range(31,  0) = uram_row_right;
        merged_uram_data.range(63,  32) = uram_row_left;
        one_write_burst.data = merged_uram_data;
        write_to_stream(little_pipe_tmp_prop_stm, one_write_burst);
    }
    DEBUG_PRINTF("little kernel mergeWriteResults module exits!\n");
}
#endif  // __ACC_GATHER_H__