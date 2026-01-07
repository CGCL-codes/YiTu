#pragma once
// #include "helpers.h"
#include "parallel.h"
#include <immintrin.h>

#define GET_NTH_BIT(x, n) ((x >> n) & 1U)
#define SET_NTH_BIT(x, n) (x |= (1U << n))
#define FLIP_NTH_BIT(x, n) (x ^= (1U << n))

inline bool bit_array_get(uint32_t *array, uint64_t i) {
  return GET_NTH_BIT(array[i/32], i%32);
}
inline void bit_array_prefetch(uint32_t *array, uint64_t i) {
  __builtin_prefetch(&array[i/32]);
}
inline void bit_array_set(uint32_t *array, uint64_t i) {
  SET_NTH_BIT(array[i/32], i%32);
}
inline void bit_array_flip(uint32_t *array, uint64_t i) {
  FLIP_NTH_BIT(array[i/32], i%32);
}

inline uint64_t bit_array_size(uint64_t size) {
    if (size == 0) {
      return 0;
    }
    if (size < 32) {
      size = 32;
    }
    uint64_t n = size / 32;
    if (n*32 < size) {
      n += 1;
    }
    return n * 4;

}
uint64_t bit_array_count(uint32_t *array, uint64_t length) {
  uint64_t count = 0;
  for (uint64_t i = 0; i < length/32; i++) {
    count += __builtin_popcount(array[i]);
  }
  return count;
}

bool is_not_empty(uint32_t *array, uint64_t length) {
  //uint64_t count = 0;
  for (uint64_t i = 0; i < length/32; i++) {
    if (array[i] > 0)
			return true;
  }
  return false;
}

class BitArray {
  public:
  uint32_t *array;
  uint64_t length;
  bool to_free;
  BitArray(uint32_t *arr, uint64_t size) {
    array = arr;
    length = size;
    to_free = false;
  }
  BitArray(uint64_t size) {
    uint64_t n = bit_array_size(size);
    array = (uint32_t *) malloc(n);
    memset(array, 0, n);
    length = n*8;
    to_free = true;
  }
  BitArray(const BitArray &other) {
    length = other.length;
    array = (uint32_t *) malloc(length/8);
    to_free = true;
    //memcpy(array, other.array, length/8);
    parallel_for(uint64_t i = 0; i < length/32; i++) {
      array[i] = other.array[i];
    }
  }

  ~BitArray() {
    if (to_free) {
      free(array);
    }
  }
  bool get(uint64_t i) {
    return bit_array_get(array, i);
  }
  void prefetch(uint64_t i) {
    return bit_array_prefetch(array, i);
  }
  void set(uint64_t i) {
    bit_array_set(array, i);
  }
  void flip(uint64_t i) {
    bit_array_flip(array, i);
  }
  uint64_t count() {
    return bit_array_count(array, length);
  }
  bool not_empty() {
    return is_not_empty(array, length);
  }
  template <class F> void map(F &f) {
    parallel_for_256 (uint64_t i = 0; i < length; i++) {
      if (get(i)) {
        f.update(i);
      }
    }
  }
};
