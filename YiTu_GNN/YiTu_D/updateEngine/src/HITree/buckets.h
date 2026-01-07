#ifndef BUCKET_H
#define BUCKET_H

#include "iterator.h"
#include "common.h"

namespace hitreespace {

template<typename KT, typename VT>
class Bucket {
typedef std::pair<KT, VT> KVT;
public:
  KVT* data_;
  uint8_t size_;

public:
  Bucket() : data_(nullptr), size_(0) { }

  Bucket(const KVT* kvs, uint32_t size, const uint8_t capacity) 
        : size_(size) {
    data_ = new KVT[capacity];
    for (uint32_t i = 0; i < size; ++ i) {
      data_[i] = kvs[i];
    }
  }

  ~Bucket() {
    if (data_ != nullptr) {
      delete[] data_;
      data_ = nullptr;
    }
  }

  inline uint8_t size() const { return size_; }

  ResultIterator<KT, VT> find(KT key) {
    for (uint32_t i = 0; i < size_; ++ i) {
      if (compare(data_[i].first, key)) {
        return {&data_[i]};
      }
    }
    return {};
  }

  bool update(KVT kv) {
    for (uint8_t i = 0; i < size_; ++ i) {
      if (compare(data_[i].first, kv.first)) {
        data_[i] = kv;
        return true;
      }
    }
    return false;
  }

  uint32_t remove(KT key) {
    bool copy = false;
    for (uint8_t i = 0; i < size_; ++ i) {
      if (compare(data_[i].first, key)) {
        copy = true;
      }
      if (copy && i + 1 < size_) {
        data_[i] = data_[i + 1];
      }
    }
    if (copy) {
      size_ --;
      return 1;
    } else {
      return 0;
    }
  }

  bool insert(KVT kv, const uint8_t capacity) {
    for (uint32_t i = 0; i < size_; ++i) {
      if (compare(data_[i].first, kv.first)) {
        return true;
      }
    }
    if (size_ < capacity) {
      data_[size_] = kv;
      size_ ++;
      return true;
    } else {
      return false;
    }
  }
};

}

#endif
