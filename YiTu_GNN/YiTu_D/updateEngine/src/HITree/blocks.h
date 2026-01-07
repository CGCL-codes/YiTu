#ifndef BLOCK_H
#define BLOCK_H

#include "iterator.h"
#include "common.h"

namespace hitreespace {

// #define KT uint32_t
template<typename KT>
inline void make_block(KT* data, const KT* ks, uint32_t size) {
  data[0] = size;
  for (uint32_t i = 0; i < size; ++ i) {
    data[i+1] = ks[i];
  }
}

template<typename KT>
bool block_find(KT* data, KT key) {
  for (uint32_t i = 0; i < data[0]; ++ i) {
    if (compare(data[i+1], key)) {
      return true;
    }
  }
  return false;
}

template<typename KT>
bool block_insert(KT* data, KT k) {
  uint32_t size = data[0];
  KT* data_ = data+1;
  for (uint32_t i = 0; i < size; ++i) {
    if (compare(data_[i], k)) {
      return true;
    }
  }
  uint32_t capacity = (GRAIN / sizeof(KT)) - 1;
  if (size < capacity) {
    data_[size] = k;
    data[0] ++;
    return true;
  } else {
    return false;
  }
}

template<typename KT>
uint32_t block_remove(KT* data, KT k) {
  bool copy = false;
  uint32_t size = data[0];
  KT* data_ = data+1;
  for (uint32_t i = 0; i < size; ++ i) {
    if (compare(data_[i], k)) {
      copy = true;
    }
    if (copy && i + 1 < size) {
      data_[i] = data_[i + 1];
    }
  }
  if (copy) {
    data[0] --;
    return 1;
  } else {
    return 0;
  }
}

template<typename KT>
bool block_insert_sort(KT* data, KT k) {
  uint32_t capacity = (GRAIN / sizeof(KT)) - 1;

  uint32_t size = data[0];
  KT* data_ = data+1;
  // must use int instead of uint32_t
  for (int i = 0; i < size; ++i) {
    if (compare(data_[i], k)) {
      return true;
    } else if (data_[i] > k) {
      if (size >= capacity) return false;
      // printf("%d %d\n", i, size);
      for (int j = size-1; j >= i; --j){
        data_[j+1] = data_[j];
      } 
      data_[i] = k;
      data[0]++;
      return true;
    }
  }

  if (size < capacity) {
    data_[size] = k;
    data[0] ++;
    return true;
  } else {
    return false;
  }
}
}

#endif
