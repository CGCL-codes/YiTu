#include <hls_stream.h>
#include <string.h>
#include "acc_config.h"
#include "l1_api.h"
#include "acc_data_types.h"
#include <ap_axi_sdata.h>

#include "acc_apply.h"

/*
Apply阶段的顶层内核函数
三个主要处理阶段：
1.merge_big_little_writes - 合并大小流水线数据
2.Apply - 核心属性更新逻辑
3.write_out - 最终属性写入
*/
extern "C" {
    void kernelApply(
#if HAVE_VERTEX_PROP
        ap_uint<512>                    *vertex_prop,
#endif
#if HAVE_APPLY_OUTDEG
		ap_uint<512>                    *outdegree,
#endif
		ap_uint<32>  					num_dense_paritions,
		ap_uint<32>  					num_sparse_paritions,
		unsigned int                    arg_reg,
#if LITTLE_KERNEL_NUM
		hls::stream<write_burst_pkt>    &l_merged_prop_stm,
#endif
#if BIG_KERNEL_NUM
		hls::stream<write_burst_pkt>    &b_merged_prop_stm,
#endif
        hls::stream<write_burst_pkt>    &prop_write_burst_stm
    )
{

#if HAVE_VERTEX_PROP
	#pragma HLS INTERFACE m_axi port=vertex_prop offset=slave bundle = gmem1
	#pragma HLS INTERFACE s_axilite port=vertex_prop  bundle=control
#endif

#if HAVE_APPLY_OUTDEG
	#pragma HLS INTERFACE m_axi port=outdegree offset=slave bundle = gmem2
	#pragma HLS INTERFACE s_axilite port=outdegree  bundle=control
#endif

	#pragma HLS INTERFACE s_axilite port=num_dense_paritions  bundle=control
	#pragma HLS INTERFACE s_axilite port=num_sparse_paritions  bundle=control
	#pragma HLS INTERFACE s_axilite port=return  bundle=control


#pragma HLS DATAFLOW

// 	hls::stream<ap_uint<512>>      l_write_burst_stm;
// #pragma HLS stream variable=l_write_burst_stm depth=2

	hls::stream<write_burst_dt>      write_burst_stm;
//	#pragma HLS stream variable=write_burst_stm depth=16
	hls::stream<write_burst_dt>      write_burst_stm_apply;
			
	merge_big_little_writes(
#if LITTLE_KERNEL_NUM
        		l_merged_prop_stm,
#endif
#if BIG_KERNEL_NUM
				b_merged_prop_stm,
#endif
        		write_burst_stm,
				num_dense_paritions,
				num_sparse_paritions
				);

    Apply(
#if HAVE_VERTEX_PROP
		vertex_prop, 
#endif
#if HAVE_APPLY_OUTDEG
		outdegree,
#endif
		arg_reg,
		write_burst_stm, 
		write_burst_stm_apply, 
		prop_write_burst_stm
		);

	write_out(
#if HAVE_VERTEX_PROP
		vertex_prop,
#endif
		write_burst_stm_apply
		);
}
}
