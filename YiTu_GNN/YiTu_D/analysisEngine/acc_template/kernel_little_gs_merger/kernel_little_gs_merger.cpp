#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "acc_data_types.h"
#include "fpga_application.h"
#include <ap_axi_sdata.h>

// REGRAPH_LITTLE_KERNEL_ID_TEXT
// REGRAPH_LITTLE_KERNEL_ID_ARRAY REGRAPH_LITTLE_KERNEL_ID_TEXT
// REGRAPH_LITTLE_KERNEL_ID_ARRAY
// REGRAPH_LITTLE_KERNEL_ID_TEXT

void merge_tmp_prop_little_krnls( 
	// TODO
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_1,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_2,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_3,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_4,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_5,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_6,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_7,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_8,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_9,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_10,
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_11,
				hls::stream<write_burst_pkt>      &l_write_burst_stm
				)
{

    l_tmp_prop_pkt a_little_krnl_tmp_prop_pkt[LITTLE_KERNEL_NUM];   
#pragma HLS ARRAY_PARTITION variable=a_little_krnl_tmp_prop_pkt dim=0 complete

    bool process_flag[LITTLE_KERNEL_NUM];   
#pragma HLS ARRAY_PARTITION variable=process_flag dim=0 complete

	for(int i = 0; i < LITTLE_KERNEL_NUM; i++){
	#pragma HLS unroll
		process_flag[i] = 0;
	}
	ap_uint<512> merged_write_burst;
	
	write_burst_pkt one_write_burst;

	uint inner_idx = 0;
	uint outer_idx = 0;

	merge_tmp_prop_little_krnls:
    while (true){
#pragma HLS pipeline style=flp
		
		// TODO: 
		if(!process_flag[0]) process_flag[0] = read_from_stream_nb(l_tmp_prop_stm_1, a_little_krnl_tmp_prop_pkt[0]);
		if(!process_flag[1]) process_flag[1] = read_from_stream_nb(l_tmp_prop_stm_2, a_little_krnl_tmp_prop_pkt[1]);
		if(!process_flag[2]) process_flag[2] = read_from_stream_nb(l_tmp_prop_stm_3, a_little_krnl_tmp_prop_pkt[2]);
		if(!process_flag[3]) process_flag[3] = read_from_stream_nb(l_tmp_prop_stm_4, a_little_krnl_tmp_prop_pkt[3]);
		if(!process_flag[4]) process_flag[4] = read_from_stream_nb(l_tmp_prop_stm_5, a_little_krnl_tmp_prop_pkt[4]);
		if(!process_flag[5]) process_flag[5] = read_from_stream_nb(l_tmp_prop_stm_6, a_little_krnl_tmp_prop_pkt[5]);
		if(!process_flag[6]) process_flag[6] = read_from_stream_nb(l_tmp_prop_stm_7, a_little_krnl_tmp_prop_pkt[6]);
		if(!process_flag[7]) process_flag[7] = read_from_stream_nb(l_tmp_prop_stm_8, a_little_krnl_tmp_prop_pkt[7]);
		if(!process_flag[8]) process_flag[8] = read_from_stream_nb(l_tmp_prop_stm_9, a_little_krnl_tmp_prop_pkt[8]);
		if(!process_flag[9]) process_flag[9] = read_from_stream_nb(l_tmp_prop_stm_10, a_little_krnl_tmp_prop_pkt[9]);
		if(!process_flag[10]) process_flag[10] = read_from_stream_nb(l_tmp_prop_stm_11, a_little_krnl_tmp_prop_pkt[10]);
		
		//if(!process_flag[1]) process_flag[1] = read_from_stream(l_tmp_prop_stm_2, a_little_krnl_tmp_prop_pkt[1]);
		bool merge_flag = 
						process_flag[0] &
						process_flag[1] &
						process_flag[2] &
						process_flag[3] &
						process_flag[4] &
						process_flag[5] &
						process_flag[6] &
						process_flag[7] &
						process_flag[8] &
						process_flag[9] &
						process_flag[10] &
						1;//  = process_flag[0] & process_flag[1] & process_flag[1] ... //if there are more channels
		if(merge_flag){    
			// optimize this logic latter.
			ap_uint<64> merged_ret = 0;

			uint uram_row_left = 0;
			uint uram_row_right = 0;

			for(int i = 0; i < LITTLE_KERNEL_NUM; i ++){
			#pragma HLS UNROLL
				// uram_row_left += a_little_krnl_tmp_prop_pkt[i].data.range(31, 0);
				// uram_row_right += a_little_krnl_tmp_prop_pkt[i].data.range(63,32);
				uram_row_left = PROP_COMPUTE_STAGE4(uram_row_left, a_little_krnl_tmp_prop_pkt[i].data.range(31, 0));
				uram_row_right = PROP_COMPUTE_STAGE4(uram_row_right, a_little_krnl_tmp_prop_pkt[i].data.range(63,32));
			}

			merged_ret.range(31, 0) = uram_row_left;
			merged_ret.range(63, 32) = uram_row_right;
			
			merged_write_burst.range(63 + (inner_idx << 6),  (inner_idx << 6)) = merged_ret; 

			inner_idx ++; 

			if(inner_idx == 8){
				inner_idx = 0;
				one_write_burst.data = merged_write_burst;
				write_to_stream(l_write_burst_stm, one_write_burst);
			}

			for(int i = 0; i < LITTLE_KERNEL_NUM; i++){
			#pragma HLS unroll
				process_flag[i] = 0;
			}
		}
    }
}


extern "C" {
    void  kernelLittleGSMerger(
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_1,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_2,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_3,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_4,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_5,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_6,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_7,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_8,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_9,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_10,
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_11,
		hls::stream<write_burst_pkt>      		&l_write_burst_stm
    )
{

    #pragma HLS interface ap_ctrl_none port = return

#pragma HLS DATAFLOW

	merge_tmp_prop_little_krnls(
							l_tmp_prop_stm_1,
							l_tmp_prop_stm_2,
							l_tmp_prop_stm_3,
							l_tmp_prop_stm_4,
							l_tmp_prop_stm_5,
							l_tmp_prop_stm_6,
							l_tmp_prop_stm_7,
							l_tmp_prop_stm_8,
							l_tmp_prop_stm_9,
							l_tmp_prop_stm_10,
							l_tmp_prop_stm_11,
							l_write_burst_stm
							);

}
}
