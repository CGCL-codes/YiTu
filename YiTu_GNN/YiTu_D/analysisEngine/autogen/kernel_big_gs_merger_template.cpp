#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "acc_data_types.h"
#include "fpga_application.h"
#include <ap_axi_sdata.h> 

// REGRAPH_BIG_KERNEL_ID_TEXT
// REGRAPH_BIG_KERNEL_ID_ARRAY REGRAPH_BIG_KERNEL_ID_TEXT
// REGRAPH_BIG_KERNEL_ID_ARRAY
// REGRAPH_BIG_KERNEL_ID_TEXT

void merge_tmp_prop_big_krnls ( 
	// TODO 
				hls::stream<b_tmp_prop_pkt>    			&b_tmp_prop_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%,
				hls::stream<write_burst_pkt>      		&b_write_burst_stm
				)
{
    b_tmp_prop_pkt a_big_krnl_tmp_prop_pkt[BIG_KERNEL_NUM];
#pragma HLS ARRAY_PARTITION variable=a_big_krnl_tmp_prop_pkt dim=0 complete

    bool process_flag[BIG_KERNEL_NUM];   
#pragma HLS ARRAY_PARTITION variable=process_flag dim=0 complete


	for(int i = 0; i < BIG_KERNEL_NUM; i++){
	#pragma HLS unroll
		process_flag[i] = 0;
	}

	ap_uint<512>  merged_write_burst;

	write_burst_pkt one_write_burst;

	uint outer_idx = 0;

	uint tmp_prop_arrary[16];
#pragma HLS ARRAY_PARTITION variable=tmp_prop_arrary dim=0 complete

	merge_tmp_prop_big_krnls:
    while(true)
    {
#pragma HLS pipeline style=flp
	
		// TODO
		if(!process_flag[%REGRAPH_BIG_KERNEL_ID_ARRAY%]) process_flag[%REGRAPH_BIG_KERNEL_ID_ARRAY%] = read_from_stream_nb(b_tmp_prop_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%, a_big_krnl_tmp_prop_pkt[%REGRAPH_BIG_KERNEL_ID_ARRAY%]);
		
		bool merge_flag =
						 process_flag[%REGRAPH_BIG_KERNEL_ID_ARRAY%] &
						 1;//  = process_flag[0] & process_flag[1] & process_flag[1] ... //if there are more channels
		
		if(merge_flag){  
			// optimize this logic latter.
			for(int i = 0; i < 16; i ++){
			#pragma HLS UNROLL
				tmp_prop_arrary[i] = 0;
			}

			for(int i = 0; i < BIG_KERNEL_NUM; i ++){
			#pragma HLS UNROLL
				for(int j = 0; j < 16; j ++){
				#pragma HLS UNROLL
					// tmp_prop_arrary[j] += a_big_krnl_tmp_prop_pkt[i].data.range(31 + (j << 5), (j << 5));
					uint update = a_big_krnl_tmp_prop_pkt[i].data.range(31 + (j << 5), (j << 5));
					tmp_prop_arrary[j] = PROP_COMPUTE_STAGE4(tmp_prop_arrary[j], update);
				}
			} 

			for(int i = 0; i < 16; i ++){
			#pragma HLS UNROLL
				merged_write_burst.range(31 + (i << 5), (i << 5)) = tmp_prop_arrary[i];
			}
			
			one_write_burst.data = merged_write_burst;
			write_to_stream(b_write_burst_stm, one_write_burst);
			
			for(int i = 0; i < BIG_KERNEL_NUM; i++){
			#pragma HLS unroll
				process_flag[i] = 0;
			}
		}
    }
}

extern "C" {
    void  kernelBigGSMerger(
        hls::stream<b_tmp_prop_pkt>    			&b_tmp_prop_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%,
		hls::stream<write_burst_pkt>      		&b_write_burst_stm
    )
{

    #pragma HLS interface ap_ctrl_none port = return

#pragma HLS DATAFLOW

	merge_tmp_prop_big_krnls(
							b_tmp_prop_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%,
							b_write_burst_stm
							);
}
}
