// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef SLIDING_QUEUE_H_
#define SLIDING_QUEUE_H_

#include <algorithm>
#include <string.h>
// #include "platform_atomics.h"


/*
GAP Benchmark Suite
Class:  SlidingQueue
Author: Scott Beamer

Double-buffered queue so appends aren't seen until SlideWindow() called
 - Use QueueBuffer when used in parallel to avoid false sharing by doing
   bulk appends from thread-local storage
*/


template <typename T>
class QueueBuffer;

template <typename T>
class SlidingQueue {
 public:
  T *shared;
  size_t shared_in;
  size_t shared_out_start;
  size_t shared_out_end;
  friend class QueueBuffer<T>;

  explicit SlidingQueue(size_t shared_size) {
    shared = new T[shared_size];
    reset();
  }
  explicit SlidingQueue(const SlidingQueue &other, size_t shared_size) {
    shared_in = other.shared_in;
    shared_out_start = other.shared_out_start;
    shared_out_end = other.shared_out_end;
    shared = new T[shared_size];
    //memcpy(shared, other.shared, shared_size * sizeof(T));
    parallel_for_256(uint64_t i = 0; i < shared_size; i++) {
      shared[i] = other.shared[i];
    }
  }

  ~SlidingQueue() {
    delete[] shared;
  }

  void push_back(T to_add) {
    shared[shared_in++] = to_add;
  }

  bool empty() const {
    return shared_out_start == shared_out_end;
  }

  void reset() {
    shared_out_start = 0;
    shared_out_end = 0;
    shared_in = 0;
  }

  void slide_window() {
    shared_out_start = shared_out_end;
    shared_out_end = shared_in;
  }

  typedef T* iterator;

  iterator begin() const {
    return shared + shared_out_start;
  }

  iterator end() const {
    return shared + shared_out_end;
  }

  size_t size() const {
    return end() - begin();
  }
};


template <typename T>
class QueueBuffer {
 public:
  size_t in;
  T *local_queue;
  SlidingQueue<T> &sq;
  const size_t local_size;

  explicit QueueBuffer(SlidingQueue<T> &master, size_t given_size = 16384)
      : sq(master), local_size(given_size) {
    in = 0;
    local_queue = new T[local_size];
  }

  ~QueueBuffer() {
    delete[] local_queue;
  }

  void push_back(T to_add) {
    if (in == local_size)
      flush();
    local_queue[in++] = to_add;
  }

  void flush() {
    T *shared_queue = sq.shared;
    size_t copy_start = __sync_fetch_and_add(&sq.shared_in, in);
    std::copy(local_queue, local_queue+in, shared_queue+copy_start);
    //printf("flushing %lu items\n", in);
    in = 0;
  }
};

#endif  // SLIDING_QUEUE_H_
