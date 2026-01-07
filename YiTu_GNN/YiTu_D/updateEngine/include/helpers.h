#ifndef HELPERS_H
#define HELPERS_H

#if OPENMP == 1
#include <omp.h>
#else
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_opadd.h>
#include <cilk/reducer_opxor.h>
#endif

#include <sys/time.h>
#include <stdint.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#ifdef debug
#define dprintf(fmt, args...) fprintf(stderr, fmt, ##args)
#else
#define dprintf(fmt, args...) /* Don't do anything in release builds */
#endif

#if OPENMP == 1
#define par_for _Pragma("omp parallel for") for
#else
#define par_for cilk_for
#endif




typedef uint32_t el_t;

/*
// find index of first 1-bit (least significant bit)
static inline int bsf_word(int word) {
  int result;
  __asm__ volatile("bsf %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline int bsr_word(int word) {
  int result;
  __asm__ volatile("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}
*/
/*
static int get_worker_number() {
#if OPENMP == 1
  return omp_get_thread_num();
#else
  return __cilkrts_get_worker_number();
#endif
}
static int get_nworkers() {
#if OPENMP == 1
  return omp_get_max_threads();
#else
  return __cilkrts_get_nworkers();
#endif
}
*/
/*
typedef struct _pair_int {
  uint32_t x; // length in array
  uint32_t y; // depth
} pair_int;
*/

typedef struct _pair_els {
  el_t x; 
  el_t y; 
} pair_els;

/*
typedef struct _pair_double {
  double x;
  double y;
} pair_double;
*/

// int isPowerOfTwo(int x) { return ((x != 0) && !(x & (x - 1))); }

// same as find_leaf, but does it for any level in the tree
// index: index in array
// len: length of sub-level.
// int find_node(int index, int len) { return (index / len) * len; }

inline
double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

inline float horizontal_add (__m256 a) {
    __m256 t1 = _mm256_hadd_ps(a,a);
    __m256 t2 = _mm256_hadd_ps(t1,t1);
    __m128 t3 = _mm256_extractf128_ps(t2,1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
    return _mm_cvtss_f32(t4);        
}

inline __m128i load_4_32(void * mem_addr) {
  return _mm_lddqu_si128((__m128i *)mem_addr);
}
/*
inline __m128i load_4_24(void * mem_addr) {
  __m128i data = _mm_lddqu_si128((__m128i *)mem_addr);
  __mmask16 mask = 0x777;
  __m128i idx = _mm_setr_epi8(3,15,14,13, 2,12,11,10, 1,9,8,7, 0,6,5,4);
  return _mm_maskz_permutexvar_epi8(mask, idx, data);
}
*/
inline __m128i load_4_16(void * mem_addr) {
  return _mm_lddqu_si128((__m128i *)mem_addr);
}

/*
static long get_usecs() {
    struct timeval st;
    gettimeofday(&st,NULL);
    return st.tv_sec*1000000 + st.tv_usec;
}
static double get_static_upper_density_bound(uint32_t depth, uint32_t real_logN) {
  double upper;
  upper = 3.0 / 4.0 + ((.25 * depth) / bsr_word((1U << real_logN)/(1 << (bsr_word(real_logN)))));
  if (upper > ((float) (1 << bsr_word(real_logN)) - 1)/(1 << bsr_word(real_logN))) {
    upper = (((float) (1 << bsr_word(real_logN)) - 1)/(1 << bsr_word(real_logN)))+.001;
  }
  return upper;
}
*/
inline void segfault() {
  uint64_t x = 0;
  uint64_t *y = (uint64_t *) x;
  *y = 0;
}

/*
#include "table.h"
double *upper_density_bound_table = (double *) table;
#define GET_UPPER_DENSITY_BOUND(depth, real_logN) upper_density_bound_table[depth*32+real_logN]

/ I'm assuming that this is between 0 and 2^31
uint64_t get_worker_num() {
  return __cilkrts_get_worker_number() + 1;
  // return 0;
}
*/
#endif
