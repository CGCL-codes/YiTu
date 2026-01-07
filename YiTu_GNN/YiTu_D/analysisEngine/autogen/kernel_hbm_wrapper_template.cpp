#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "acc_data_types.h"
#include <ap_axi_sdata.h>

#include "hbm_wrapper.h"

// REGRAPH_ALL_KERNEL_ID_TEXT
// REGRAPH_LITTLE_KERNEL_ID_TEXT
// REGRAPH_BIG_KERNEL_ID_TEXT
// REGRAPH_ALL_KERNEL_ID_TEXT REGRAPH_ALL_KERNEL_ID_TEXT
// REGRAPH_LITTLE_KERNEL_ID_ARRAY REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT REGRAPH_LITTLE_KERNEL_ID_TEXT
// REGRAPH_BIG_KERNEL_ID_ARRAY REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT REGRAPH_BIG_KERNEL_ID_TEXT
// REGRAPH_ALL_KERNEL_ID_TEXT

extern "C" {
    void  kernelHBMWrapper(
        ap_uint<512>                    *src_prop_%REGRAPH_ALL_KERNEL_ID_TEXT%,


        ap_uint<512>                    *new_prop_%REGRAPH_ALL_KERNEL_ID_TEXT%,


		uint num_dense_paritions,
		uint num_sparse_paritions,


        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%,


        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%,


        hls::stream<b_cacheline_request_pkt>    &b_cacheline_request_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%,


        hls::stream<b_cacheline_response_pkt>   &b_cacheline_response_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%,

		hls::stream<write_burst_pkt>      		&prop_write_burst_stm

    )
{

	#pragma HLS INTERFACE m_axi port=src_prop_%REGRAPH_ALL_KERNEL_ID_TEXT% offset=slave bundle = gmem%REGRAPH_ALL_KERNEL_ID_TEXT%
	
	#pragma HLS INTERFACE m_axi port=new_prop_%REGRAPH_ALL_KERNEL_ID_TEXT% offset=slave bundle = gmem%REGRAPH_ALL_KERNEL_ID_TEXT%

	#pragma HLS INTERFACE s_axilite port=src_prop_%REGRAPH_ALL_KERNEL_ID_TEXT%  bundle=control
	
	#pragma HLS INTERFACE s_axilite port=new_prop_%REGRAPH_ALL_KERNEL_ID_TEXT%  bundle=control

	#pragma HLS INTERFACE s_axilite port=num_dense_paritions  bundle=control
	#pragma HLS INTERFACE s_axilite port=num_sparse_paritions  bundle=control
	#pragma HLS INTERFACE s_axilite port=return  bundle=control



#pragma HLS DATAFLOW

	hls::stream<write_burst_dt>      write_burst_stm;
#pragma HLS stream variable=write_burst_stm depth=2

	hls::stream<write_burst_dt>      l_write_burst_stm_local;
#pragma HLS stream variable=l_write_burst_stm_local depth=2

	hls::stream<write_burst_dt>      b_write_burst_stm_local;
#pragma HLS stream variable=b_write_burst_stm_local depth=2

	littleKernelReadMemory<%REGRAPH_LITTLE_KERNEL_ID_ARRAY%>(%REGRAPH_LITTLE_KERNEL_ID_ARRAY%, src_prop_%REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT%, num_dense_paritions, l_ppb_request_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%, l_ppb_response_stm_%REGRAPH_LITTLE_KERNEL_ID_TEXT%);

	bigKernelReadMemory<%REGRAPH_BIG_KERNEL_ID_ARRAY%>(%REGRAPH_BIG_KERNEL_ID_ARRAY%, src_prop_%REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT%, num_sparse_paritions, b_cacheline_request_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%, b_cacheline_response_stm_%REGRAPH_BIG_KERNEL_ID_TEXT%);


	write_out(
			new_prop_%REGRAPH_ALL_KERNEL_ID_TEXT%,
			prop_write_burst_stm			
		);
}
}
