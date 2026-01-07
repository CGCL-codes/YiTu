#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "l2.h"
#include "acc_data_types.h"
#include <ap_axi_sdata.h>

//将小流水线(处理低度数顶点)的临时属性数据合并为512位宽突发写入
void uramrow2writeburst(
				hls::stream<l_tmp_prop_pkt>      	&l_merged_prop_stm,
        		hls::stream<ap_uint<512>>  			&write_burst_stm,
				uint num_dense_paritions
				)
{

    l_tmp_prop_pkt a_little_krnl_tmp_prop_pkt;   

	uint total_num_write_burst = (LITTLE_KERNEL_DST_BUFFER_SIZE >> 4) * num_dense_paritions;

	ap_uint<512> merged_write_burst;

	uint inner_idx = 0;

	merge_tmp_prop_little_krnls:
    while (true){
#pragma HLS PIPELINE II=1

		if(!total_num_write_burst) 
		{
			break;
		}

		read_from_stream(l_merged_prop_stm, a_little_krnl_tmp_prop_pkt);
		
		// optimize this logic latter.
		ap_uint<64> merged_ret = 0;

		merged_ret.range(31, 0) = a_little_krnl_tmp_prop_pkt.data.range(31, 0);;
		merged_ret.range(63, 32) = a_little_krnl_tmp_prop_pkt.data.range(63,32);
		
        merged_write_burst.range(63 + (inner_idx << 6),  (inner_idx << 6)) = merged_ret; 

		inner_idx ++; 
		
		if(inner_idx == 8){
			inner_idx = 0;
        	write_to_stream(write_burst_stm, merged_write_burst);
			total_num_write_burst --;
		} 
    }
}


//合并大/小流水线的写入请求
void merge_big_little_writes(
#if LITTLE_KERNEL_NUM
				hls::stream<write_burst_pkt>      		&l_write_burst_stm,
#endif
#if BIG_KERNEL_NUM
				hls::stream<write_burst_pkt>      		&b_write_burst_stm,
#endif
        		hls::stream<write_burst_dt>  			&write_burst_stm,
				uint num_dense_paritions,
				uint num_sparse_paritions
				)
{

	write_burst_pkt one_little_krnl_tmp_prop_write_burst;
	write_burst_pkt one_big_krnl_tmp_prop_write_burst;

	uint total_num_write_burst = ((LITTLE_KERNEL_DST_BUFFER_SIZE >> 4) * num_dense_paritions + (BIG_KERNEL_DST_BUFFER_SIZE >> 4) * num_sparse_paritions);
	
	uint outer_idx_little_krnl = 0;
	uint outer_idx_big_krnl = ((LITTLE_KERNEL_DST_BUFFER_SIZE * num_dense_paritions) >> 4);

	merge_big_little_writes:
    while(true){
#pragma HLS PIPELINE II=1	

		if(!total_num_write_burst){
			write_burst_dt  one_write_burst;
			one_write_burst.end_flag = 1; 
        	write_to_stream(write_burst_stm, one_write_burst);
			DEBUG_PRINTF("[INFO] merge_big_little_writes exits \n"); 			
			break;
		} 

#if LITTLE_KERNEL_NUM
		if(read_from_stream_nb(l_write_burst_stm, one_little_krnl_tmp_prop_write_burst)){
			write_burst_dt  one_write_burst;
			one_write_burst.write_idx = outer_idx_little_krnl ++;
			one_write_burst.data = one_little_krnl_tmp_prop_write_burst.data;
			one_write_burst.end_flag = 0; 
        	write_to_stream(write_burst_stm, one_write_burst);
			total_num_write_burst --;
		}
#endif
#if BIG_KERNEL_NUM
#if LITTLE_KERNEL_NUM
		else if(read_from_stream_nb(b_write_burst_stm, one_big_krnl_tmp_prop_write_burst)){
#else
			if(read_from_stream_nb(b_write_burst_stm, one_big_krnl_tmp_prop_write_burst)){
#endif
				write_burst_dt  one_write_burst;
				one_write_burst.write_idx = outer_idx_big_krnl ++;
				one_write_burst.data = one_big_krnl_tmp_prop_write_burst.data;
				one_write_burst.end_flag = 0; 
				write_to_stream(write_burst_stm, one_write_burst);
				total_num_write_burst --;
			}
#endif
	}
}

//应用阶段核心处理逻辑，包含用户定义的applyFunc()
void Apply(	
#if HAVE_VERTEX_PROP
			ap_uint<512>                   			*vertex_prop,
#endif
#if HAVE_APPLY_OUTDEG
			ap_uint<512>                   			*outdegree,
#endif
			unsigned int                            arg_reg,
			hls::stream<write_burst_dt>      		&write_burst_stm_in,
			hls::stream<write_burst_dt>      		&write_burst_stm_out,
			hls::stream<write_burst_pkt>  			&write_burst_pkt_stm
				)
	{
//#pragma HLS dependence variable=vertex_prop inter false
//#pragma HLS dependence variable=vertex_prop intra false
	write_burst_pkt  one_write_burst_pkt;
	write_burst_dt   one_write_burst_out;

	Apply:
	while(true){
	#pragma HLS PIPELINE II=1	
		
		write_burst_dt   one_write_burst;
		read_from_stream(write_burst_stm_in, one_write_burst);

		if(one_write_burst.end_flag){

			one_write_burst_out.end_flag = 1;
			write_to_stream(write_burst_stm_out, one_write_burst_out);

			one_write_burst_pkt.last = 1; 
			write_to_stream(write_burst_pkt_stm, one_write_burst_pkt);
			break;
		} 

		uint write_idx = one_write_burst.write_idx;

		ap_uint<512> tmp_prop = one_write_burst.data;
#if HAVE_VERTEX_PROP
		ap_uint<512> vprop = vertex_prop[write_idx];
#else
		ap_uint<512> vprop = 0;
#endif
#if HAVE_APPLY_OUTDEG
		ap_uint<512> outDeg = outdegree[write_idx];
#else
		ap_uint<512> outDeg = 0;
#endif

		ap_uint<512> new_prop;

		// #define prop_t int
		for(int u = 0; u < 16; u++){
		#pragma HLS unroll

			prop_t update =   applyFunc(
										tmp_prop.range(31 + (u << 5), (u << 5)),
										vprop.range(31 + (u << 5), (u << 5)),
										outDeg.range(31 + (u << 5), (u << 5)),
										arg_reg
										);

			new_prop.range(31 + (u << 5), (u << 5)) = update;
		}

#if HAVE_VERTEX_PROP
		/* update the property for the APPLY first*/
		//vertex_prop[write_idx] = new_prop;
		one_write_burst_out.data = new_prop;
		one_write_burst_out.write_idx = write_idx;
		one_write_burst_out.end_flag = 0;
		write_to_stream(write_burst_stm_out, one_write_burst_out);
#endif

		/* for all other channels */
		one_write_burst_pkt.data = new_prop;
		one_write_burst_pkt.dest = write_idx;
		one_write_burst_pkt.last = 0;
		write_to_stream(write_burst_pkt_stm, one_write_burst_pkt);
	}
}

//将最终属性值写入内存
void write_out (
#if HAVE_VERTEX_PROP
				ap_uint<512>                   	*vertex_prop,
#endif
	        	hls::stream<write_burst_dt>  	&write_burst_stm
				)
{		
	uint write_idx = 0;

	write_out:
    while(true){
#pragma HLS PIPELINE II=1	
		
		write_burst_dt  one_write_burst;
		
		read_from_stream(write_burst_stm, one_write_burst);

		if(one_write_burst.end_flag){
			break;
		} 

		write_idx = one_write_burst.write_idx;

		ap_uint<512> new_prop = one_write_burst.data;
		// TODO
#if HAVE_VERTEX_PROP
		vertex_prop[write_idx] = new_prop;
#endif
	}
}
