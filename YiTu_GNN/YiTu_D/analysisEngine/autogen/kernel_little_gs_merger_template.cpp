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
				hls::stream<l_tmp_prop_pkt>    	  &l_tmp_prop_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%,
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
		if(!process_flag[%REGRAPH_LITTLE_KERNEL_ID_ARRAY%]) process_flag[%REGRAPH_LITTLE_KERNEL_ID_ARRAY%] = read_from_stream_nb(l_tmp_prop_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%, a_little_krnl_tmp_prop_pkt[%REGRAPH_LITTLE_KERNEL_ID_ARRAY%]);
		
		//if(!process_flag[1]) process_flag[1] = read_from_stream(l_tmp_prop_stm_2, a_little_krnl_tmp_prop_pkt[1]);
		bool merge_flag = 
						process_flag[%REGRAPH_LITTLE_KERNEL_ID_ARRAY%] &
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
        hls::stream<l_tmp_prop_pkt>    			&l_tmp_prop_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%,
		hls::stream<write_burst_pkt>      		&l_write_burst_stm
    )
{

    #pragma HLS interface ap_ctrl_none port = return

#pragma HLS DATAFLOW

	merge_tmp_prop_little_krnls(
							l_tmp_prop_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%,
							l_write_burst_stm
							);

}
}
