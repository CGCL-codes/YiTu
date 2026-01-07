#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "acc_data_types.h"
#include <ap_axi_sdata.h>

// REGRAPH_ALL_KERNEL_ID_TEXT

template <int an_unused_template_parameter>
void littleKernelReadMemory(
				int i,
        		ap_uint<512>                    	*src_prop,
				uchar 								num_paritions,
                hls::stream<l_ppb_request_pkt>    	&l_ppb_request_stm,
                hls::stream<l_ppb_response_pkt>   	&l_ppb_response_stm
                ){ 

#pragma HLS function_instantiate variable=i

    l_ppb_request_pkt one_ppb_request_pkg;
    l_ppb_response_pkt one_ppb_response_pkg;

	uchar processed_num_partitions = 0;

	ppb_response_dt one_ppb_response;
    
	littleKernelReadMemory:
    while(true){
#pragma HLS PIPELINE //II=1

        if(read_from_stream_nb(l_ppb_request_stm, one_ppb_request_pkg)){

			ppb_request_dt one_ppb_request;
			one_ppb_request.request_round = one_ppb_request_pkg.data;
			one_ppb_request.end_flag = one_ppb_request_pkg.last;

			ap_uint<32>  base_addr = one_ppb_request.request_round << LOG2_SRC_BUFFER_SIZE >> 4;

			if(one_ppb_request.end_flag){
				
				one_ppb_response.end_flag = one_ppb_request.end_flag;
				one_ppb_response_pkg.last = one_ppb_response.end_flag;
				write_to_stream(l_ppb_response_stm, one_ppb_response_pkg);
				processed_num_partitions ++;
				DEBUG_PRINTF("processed_num_partitions %d, %d !\n", processed_num_partitions, num_paritions);
			}
			else{
				
				for(int i = 0; i < (SRC_BUFFER_SIZE >> 4); i ++){
					int addr = base_addr + i;
					one_ppb_response.addr = addr;
					one_ppb_response.data = src_prop[addr];
					one_ppb_response.end_flag = one_ppb_request.end_flag;
					
					one_ppb_response_pkg.data = one_ppb_response.data;
					one_ppb_response_pkg.dest = one_ppb_response.addr;
					one_ppb_response_pkg.last = one_ppb_response.end_flag;
					write_to_stream(l_ppb_response_stm, one_ppb_response_pkg);
				}
			}
		}

		if(processed_num_partitions == num_paritions){
			DEBUG_PRINTF("HBMwrapperReadMemory exists %d!\n", processed_num_partitions);
			break;
		}
    }
}

template <int an_unused_template_parameter>
void bigKernelReadMemory(
				int i,
        		ap_uint<512>                    	 	*src_prop,
				uchar 								 	num_paritions,
                hls::stream<b_cacheline_request_pkt>    &b_cacheline_request_stm,
                hls::stream<b_cacheline_response_pkt>   &b_cacheline_response_stm
                ){ 

#pragma HLS function_instantiate variable=i

    b_cacheline_request_pkt one_cacheline_request_pkg;
    b_cacheline_response_pkt one_cacheline_response_pkg;

    ap_uint<27> last_requested_cacheline_idx = -1;
    ap_uint<512> last_requested_cacheline;

	uchar processed_num_partitions = 0;

	cacheline_response_dt one_cacheline_response;

    bigKernelReadMemory:
    while(true){
#pragma HLS PIPELINE II=1

        bool process_flag = read_from_stream_nb(b_cacheline_request_stm, one_cacheline_request_pkg);

		cacheline_dt one_cacheline_request;
		one_cacheline_request.idx = one_cacheline_request_pkg.data;
		one_cacheline_request.dst = one_cacheline_request_pkg.dest;
		one_cacheline_request.end_flag = one_cacheline_request_pkg.last;

        if(process_flag){

			if(one_cacheline_request.end_flag){
				one_cacheline_response.data = 0;
				last_requested_cacheline_idx = -1;
				processed_num_partitions ++;
				DEBUG_PRINTF("processed_num_partitions %d!\n", processed_num_partitions);
			}
			else{

				if(one_cacheline_request.idx == last_requested_cacheline_idx){
                	one_cacheline_response.data = last_requested_cacheline;
				}
				else{
					one_cacheline_response.data = src_prop[one_cacheline_request.idx];

					last_requested_cacheline_idx = one_cacheline_request.idx;
					last_requested_cacheline = one_cacheline_response.data ;  
				}
			}
			
			one_cacheline_response.end_flag = one_cacheline_request.end_flag;
			one_cacheline_response.dst = one_cacheline_request.dst;

			one_cacheline_response_pkg.data = one_cacheline_response.data;
			one_cacheline_response_pkg.dest = one_cacheline_response.dst;
			one_cacheline_response_pkg.last = one_cacheline_response.end_flag;
            write_to_stream(b_cacheline_response_stm, one_cacheline_response_pkg);
        }

		if(processed_num_partitions == num_paritions) break;
    }
}


void write_out (
	// TODO
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
        		hls::stream<write_burst_pkt>  	&write_burst_stm
				)
{		
	uint write_idx = 0;

	write_out:
    while(true){
#pragma HLS PIPELINE II=1	
		
		write_burst_pkt  one_write_burst;
		
		if(read_from_stream_nb(write_burst_stm, one_write_burst)){

			write_idx = one_write_burst.dest;

			if(one_write_burst.last){
				DEBUG_PRINTF("write_out exits \n"); 			
				break;
			} 

			ap_uint<512> new_prop = one_write_burst.data;
			// TODO
			new_prop_1[write_idx] = new_prop;
			new_prop_2[write_idx] = new_prop;
			new_prop_3[write_idx] = new_prop;
			new_prop_4[write_idx] = new_prop;
			new_prop_5[write_idx] = new_prop;
			new_prop_6[write_idx] = new_prop;
			new_prop_7[write_idx] = new_prop;
			new_prop_8[write_idx] = new_prop;
			new_prop_9[write_idx] = new_prop;
			new_prop_10[write_idx] = new_prop;
			new_prop_11[write_idx] = new_prop;
			new_prop_12[write_idx] = new_prop;
			new_prop_13[write_idx] = new_prop;
			new_prop_14[write_idx] = new_prop;
		}
	}
}
