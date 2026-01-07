#ifndef ITERATOR_H
#define ITERATOR_H

#include "common.h"

namespace hitreespace {

template<typename KT, typename VT>
class ResultIterator {
typedef std::pair<KT, VT> KVT;
private:
  KVT* kv_;

public:
  ResultIterator() : kv_(nullptr) { }

  ResultIterator(KVT* kv) : kv_(kv) { }

  bool is_end() { return kv_ == nullptr; }

  KT key() { return kv_->first; }

  VT value() { return kv_->second; }

  VT* value_addr() { return &kv_->second; }

  KVT* kv() { return kv_; }

  ResultIterator<KT, VT>& operator=(const ResultIterator<KT, VT>& other) {
    if (this != &other) {
      kv_ = other.kv_;
    }
    return *this;
  }
};

}
#endif
