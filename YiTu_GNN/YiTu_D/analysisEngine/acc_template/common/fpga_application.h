//定义了图算法的应用层接口宏
#ifndef __FPGA_APPLICATION_H__
#define __FPGA_APPLICATION_H__



#include "l2.h"


#ifndef IS_ACTIVE_VERTEX
#define IS_ACTIVE_VERTEX(a)			(1)
#endif

/* source vertex property process */
#define PROP_COMPUTE_STAGE0(srcProp)                preprocessProperty(srcProp)

/* source vertex property & edge property */
#define PROP_COMPUTE_STAGE1(srcProp, edgeProp)      scatterFunc(srcProp,edgeProp)

/* destination property update in RAW solver */
#define PROP_COMPUTE_STAGE2(ori, update)            gatherFunc(ori ,update)

/* destination property update dst buffer update */
#define PROP_COMPUTE_STAGE3(ori,update)             gatherFunc(ori, update)

/* destination property merge */
#define PROP_COMPUTE_STAGE4(ori,update)             gatherFunc(ori, update)


#endif /* __FPGA_APPLICATION_H__ */
