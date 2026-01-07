
#ifndef _UTIL_H_
#define _UTIL_H_

#include <iostream>
#include <cstring>
#include <vector>
#include <cassert>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
// #include <ittnotify.h>

#include "spdlog/spdlog.h"

#ifdef DEBUG_MODE
#define PRINT_DEBUG 1
#else
#define PRINT_DEBUG 0
#endif

#define SDSL_BITVECTOR_BLOCK_SIZE 127
extern std::shared_ptr<spdlog::logger> console;

#define DEBUG(x) do { \
	if (PRINT_DEBUG) { std::cerr << x << std::endl; } \
} while (0)

#define ERROR(x) do { \
	{ std::cerr << x << std::endl; } \
} while (0)

#define PRINT(x) do { \
	{ std::cout << x << std::endl; } \
} while (0)

#if PRINT_DEBUG == 1
//#define dprintf(fmt, args...) fprintf(stderr, fmt, ##args)
#define dprintf(fmt, args...) /* Don't do anything in release builds */
#else
#define dprintf(fmt, args...) /* Don't do anything in release builds */
#endif

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
static inline uint32_t bsr_word(uint32_t word) {
  uint32_t result;
  __asm__ volatile("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}
static inline uint64_t bsr_word(uint64_t word) {
  uint64_t result;
  __asm__ volatile("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}

typedef struct _pair_int {
  uint32_t x; // length in array
  uint32_t y; // depth
} pair_int;
typedef struct _pair_uint {
  uint32_t x; // length in array
  uint32_t y; // depth
} pair_uint;

// three tuple uint
struct trip_uint {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} ;

namespace graphstore {
	float cal_time_elapsed(struct timeval* start, struct timeval* end);
	/* Print elapsed time using the start and end timeval */
	void print_time_elapsed(std::string desc, struct timeval* start, struct
													timeval* end);
	std::vector<uint32_t> get_random_permutation(uint32_t num);
}
#endif
