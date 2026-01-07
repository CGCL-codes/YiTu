//定义了加速器的全局配置参数和调试设置
#ifndef __ACC_CONFIG_H__
#define __ACC_CONFIG_H__

#define NUM_EDGE_PER_BURST 8

#define LOG2_NUM_EDGE_PER_BURST 3

#define SCATTER_PE_NUM      8
#define GATHER_PE_NUM     8
#define LOG2_GATHER_PE_NUM    3 
#define REDUCTIONTREE_NUM     8
#define EDGE_NUM            8
#define DATA_WIDTH              (512)


#define HASH_MASK  (REDUCTIONTREE_NUM - 1) 

#define ENDFLAG                                 0xffffffff

#define V_IDX_TYPE ap_uint<32>
#define L 3  // the distance used for RAW in the gather PE.


//#define SW_DEBUG

#ifdef SW_DEBUG

#include "stdio.h"
#define DEBUG_PRINTF(fmt,...)   printf(fmt,##__VA_ARGS__); fflush(stdout);

#else

#define DEBUG_PRINTF(fmt,...)   ;

#endif

#endif /* __ACC_CONFIG_H__ */