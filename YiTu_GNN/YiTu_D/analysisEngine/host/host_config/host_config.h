#ifndef __HOST_CONFIG_H__
#define __HOST_CONFIG_H__

#include <stdint.h>

// #define __ALIGN 8
// int RoundUP8 (int num){
//     return ((num + (__ALIGN - 1)) & ~(__ALIGN));
// }

#if HAVE_UNSIGNED_PROP
typedef  unsigned int       prop_t;
#else
typedef  int               prop_t;
#endif

#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)


#define ENDFLAG                 0xffffffff

#define NUM_VERTEX_ALIGNED      (((num_vertex - 1)/PARTITION_SIZE + 1) * PARTITION_SIZE)
#define MAX_NUM_PARTITION       ((num_vertex + PARTITION_SIZE - 1) / PARTITION_SIZE)


#define PROFILE


#if 1

#define DEBUG_PRINTF(fmt,...)   printf(fmt,##__VA_ARGS__); fflush(stdout);

#else

#define DEBUG_PRINTF(fmt,...)   ;

#endif

#endif /* __HOST_CONFIG_H__ */



