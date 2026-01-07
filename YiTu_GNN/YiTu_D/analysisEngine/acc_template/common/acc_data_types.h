//定义了框架中使用的主要数据结构
#ifndef __ACC_DATA_TYPES_H__
#define __ACC_DATA_TYPES_H__

#include <stdint.h>
#include "acc_config.h"
#include <ap_int.h>
#include "ap_axi_sdata.h"

typedef ap_axiu<64, 0, 0, 0> l_tmp_prop_pkt;
typedef ap_axiu<512, 0, 0, 0> b_tmp_prop_pkt;

typedef ap_axiu<512, 0, 0, 32> write_burst_pkt;

typedef ap_axiu<32, 0, 0, 0>   l_ppb_request_pkt;
typedef ap_axiu<512, 0, 0, 32> l_ppb_response_pkt;

typedef ap_axiu<32, 0, 0, 8>  b_cacheline_request_pkt;
typedef ap_axiu<512, 0, 0, 8> b_cacheline_response_pkt;


typedef   
struct ppbRequest{
    ap_uint<32> request_round;
    bool end_flag;
    }
ppb_request_dt; 


typedef 
struct ppbResponse{
    ap_uint<32> addr;
    ap_uint<512> data;
    bool end_flag;
    }
ppb_response_dt; 


typedef 
struct CachelineResponse {
    ap_uint<8> dst;
    ap_uint<512>  data;
    bool end_flag;
    }
cacheline_response_dt; 

// typedef 
// struct CachelineRequest {
//     ap_uint<26> idx;
//     ap_uint<8>  dst;
//     bool end_flag;
//     }
// cacheline_request_dt_l; 

typedef 
struct writeBurst {
    ap_uint<26> write_idx;
    ap_uint<512>  data;
    bool end_flag;
    }
write_burst_dt; 


typedef 
struct EdgeTuple {
    V_IDX_TYPE src;
    V_IDX_TYPE dst; 
    } 
edge_tuple_dt;

typedef 
struct EdgeReadBurst {
    edge_tuple_dt edges[NUM_EDGE_PER_BURST];
    }
edge_burst_dt;

typedef 
struct SrcBurst {
    V_IDX_TYPE src[NUM_EDGE_PER_BURST];
    }
src_burst_dt;

typedef 
struct MemoryRequestBurst {
    //ap_uint<26> base_idx;
    ap_uint<3>  offset;
    ap_uint<26> idx[NUM_EDGE_PER_BURST];
    bool end_flag;
    }
mem_request_burst_dt; 

typedef 
struct Cacheline {
    ap_uint<26> idx;
    ap_uint<8>  dst;
    bool end_flag;
    }
cacheline_dt; 


typedef 
struct SrcPropBurst {
    ap_uint<32> src_prop[NUM_EDGE_PER_BURST];
    }
src_prop_burst_dt;

typedef struct UpdateTuple {
    bool end_flag;
    ap_uint<32> update;
    ap_uint<20> dst;
} update_tuple_dt;

typedef 
struct UpdateTupleSet {
    update_tuple_dt tuples[GATHER_PE_NUM];
    }
update_set_dt;


#define uchar unsigned char

typedef ap_uint<8>                 ushort_raw;


typedef ap_uint<32>                 uint_raw;

typedef ap_uint<DATA_WIDTH>         uint16;

typedef ap_uint<128>                uint4_raw;

typedef ap_uint<64>                 uint_uram;



typedef struct __int2__
{
    int x;
#if HAVE_UNSIGNED_PROP
    uint_raw y;
#else
    int y;
#endif
} int2;


#endif /* __ACC_DATA_TYPES_H__ */