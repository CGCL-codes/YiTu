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
        ap_uint<512>                    *src_prop_1,
        ap_uint<512>                    *src_prop_2,
        ap_uint<512>                    *src_prop_3,
        ap_uint<512>                    *src_prop_4,
        ap_uint<512>                    *src_prop_5,
        ap_uint<512>                    *src_prop_6,
        ap_uint<512>                    *src_prop_7,
        ap_uint<512>                    *src_prop_8,
        ap_uint<512>                    *src_prop_9,
        ap_uint<512>                    *src_prop_10,
        ap_uint<512>                    *src_prop_11,
        ap_uint<512>                    *src_prop_12,
        ap_uint<512>                    *src_prop_13,
        ap_uint<512>                    *src_prop_14,


        ap_uint<512>                    *new_prop_1,
        ap_uint<512>                    *new_prop_2,
        ap_uint<512>                    *new_prop_3,
        ap_uint<512>                    *new_prop_4,
        ap_uint<512>                    *new_prop_5,
        ap_uint<512>                    *new_prop_6,
        ap_uint<512>                    *new_prop_7,
        ap_uint<512>                    *new_prop_8,
        ap_uint<512>                    *new_prop_9,
        ap_uint<512>                    *new_prop_10,
        ap_uint<512>                    *new_prop_11,
        ap_uint<512>                    *new_prop_12,
        ap_uint<512>                    *new_prop_13,
        ap_uint<512>                    *new_prop_14,


		uint num_dense_paritions,
		uint num_sparse_paritions,


        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_1,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_2,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_3,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_4,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_5,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_6,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_7,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_8,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_9,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_10,
        hls::stream<l_ppb_request_pkt>    &l_ppb_request_stm_11,


        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_1,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_2,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_3,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_4,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_5,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_6,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_7,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_8,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_9,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_10,
        hls::stream<l_ppb_response_pkt>   &l_ppb_response_stm_11,


        hls::stream<b_cacheline_request_pkt>    &b_cacheline_request_stm_1,
        hls::stream<b_cacheline_request_pkt>    &b_cacheline_request_stm_2,
        hls::stream<b_cacheline_request_pkt>    &b_cacheline_request_stm_3,


        hls::stream<b_cacheline_response_pkt>   &b_cacheline_response_stm_1,
        hls::stream<b_cacheline_response_pkt>   &b_cacheline_response_stm_2,
        hls::stream<b_cacheline_response_pkt>   &b_cacheline_response_stm_3,

		hls::stream<write_burst_pkt>      		&prop_write_burst_stm

    )
{

	#pragma HLS INTERFACE m_axi port=src_prop_1 offset=slave bundle = gmem1
	#pragma HLS INTERFACE m_axi port=src_prop_2 offset=slave bundle = gmem2
	#pragma HLS INTERFACE m_axi port=src_prop_3 offset=slave bundle = gmem3
	#pragma HLS INTERFACE m_axi port=src_prop_4 offset=slave bundle = gmem4
	#pragma HLS INTERFACE m_axi port=src_prop_5 offset=slave bundle = gmem5
	#pragma HLS INTERFACE m_axi port=src_prop_6 offset=slave bundle = gmem6
	#pragma HLS INTERFACE m_axi port=src_prop_7 offset=slave bundle = gmem7
	#pragma HLS INTERFACE m_axi port=src_prop_8 offset=slave bundle = gmem8
	#pragma HLS INTERFACE m_axi port=src_prop_9 offset=slave bundle = gmem9
	#pragma HLS INTERFACE m_axi port=src_prop_10 offset=slave bundle = gmem10
	#pragma HLS INTERFACE m_axi port=src_prop_11 offset=slave bundle = gmem11
	#pragma HLS INTERFACE m_axi port=src_prop_12 offset=slave bundle = gmem12
	#pragma HLS INTERFACE m_axi port=src_prop_13 offset=slave bundle = gmem13
	#pragma HLS INTERFACE m_axi port=src_prop_14 offset=slave bundle = gmem14
	
	#pragma HLS INTERFACE m_axi port=new_prop_1 offset=slave bundle = gmem1
	#pragma HLS INTERFACE m_axi port=new_prop_2 offset=slave bundle = gmem2
	#pragma HLS INTERFACE m_axi port=new_prop_3 offset=slave bundle = gmem3
	#pragma HLS INTERFACE m_axi port=new_prop_4 offset=slave bundle = gmem4
	#pragma HLS INTERFACE m_axi port=new_prop_5 offset=slave bundle = gmem5
	#pragma HLS INTERFACE m_axi port=new_prop_6 offset=slave bundle = gmem6
	#pragma HLS INTERFACE m_axi port=new_prop_7 offset=slave bundle = gmem7
	#pragma HLS INTERFACE m_axi port=new_prop_8 offset=slave bundle = gmem8
	#pragma HLS INTERFACE m_axi port=new_prop_9 offset=slave bundle = gmem9
	#pragma HLS INTERFACE m_axi port=new_prop_10 offset=slave bundle = gmem10
	#pragma HLS INTERFACE m_axi port=new_prop_11 offset=slave bundle = gmem11
	#pragma HLS INTERFACE m_axi port=new_prop_12 offset=slave bundle = gmem12
	#pragma HLS INTERFACE m_axi port=new_prop_13 offset=slave bundle = gmem13
	#pragma HLS INTERFACE m_axi port=new_prop_14 offset=slave bundle = gmem14

	#pragma HLS INTERFACE s_axilite port=src_prop_1  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_2  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_3  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_4  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_5  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_6  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_7  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_8  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_9  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_10  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_11  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_12  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_13  bundle=control
	#pragma HLS INTERFACE s_axilite port=src_prop_14  bundle=control
	
	#pragma HLS INTERFACE s_axilite port=new_prop_1  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_2  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_3  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_4  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_5  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_6  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_7  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_8  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_9  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_10  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_11  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_12  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_13  bundle=control
	#pragma HLS INTERFACE s_axilite port=new_prop_14  bundle=control

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

	littleKernelReadMemory<0>(0, src_prop_1, num_dense_paritions, l_ppb_request_stm_1, l_ppb_response_stm_1);
	littleKernelReadMemory<1>(1, src_prop_2, num_dense_paritions, l_ppb_request_stm_2, l_ppb_response_stm_2);
	littleKernelReadMemory<2>(2, src_prop_3, num_dense_paritions, l_ppb_request_stm_3, l_ppb_response_stm_3);
	littleKernelReadMemory<3>(3, src_prop_4, num_dense_paritions, l_ppb_request_stm_4, l_ppb_response_stm_4);
	littleKernelReadMemory<4>(4, src_prop_5, num_dense_paritions, l_ppb_request_stm_5, l_ppb_response_stm_5);
	littleKernelReadMemory<5>(5, src_prop_6, num_dense_paritions, l_ppb_request_stm_6, l_ppb_response_stm_6);
	littleKernelReadMemory<6>(6, src_prop_7, num_dense_paritions, l_ppb_request_stm_7, l_ppb_response_stm_7);
	littleKernelReadMemory<7>(7, src_prop_8, num_dense_paritions, l_ppb_request_stm_8, l_ppb_response_stm_8);
	littleKernelReadMemory<8>(8, src_prop_9, num_dense_paritions, l_ppb_request_stm_9, l_ppb_response_stm_9);
	littleKernelReadMemory<9>(9, src_prop_10, num_dense_paritions, l_ppb_request_stm_10, l_ppb_response_stm_10);
	littleKernelReadMemory<10>(10, src_prop_11, num_dense_paritions, l_ppb_request_stm_11, l_ppb_response_stm_11);

	bigKernelReadMemory<0>(0, src_prop_12, num_sparse_paritions, b_cacheline_request_stm_1, b_cacheline_response_stm_1);
	bigKernelReadMemory<1>(1, src_prop_13, num_sparse_paritions, b_cacheline_request_stm_2, b_cacheline_response_stm_2);
	bigKernelReadMemory<2>(2, src_prop_14, num_sparse_paritions, b_cacheline_request_stm_3, b_cacheline_response_stm_3);


	write_out(
			new_prop_1,
			new_prop_2,
			new_prop_3,
			new_prop_4,
			new_prop_5,
			new_prop_6,
			new_prop_7,
			new_prop_8,
			new_prop_9,
			new_prop_10,
			new_prop_11,
			new_prop_12,
			new_prop_13,
			new_prop_14,
			prop_write_burst_stm			
		);
}
}
