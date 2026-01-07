#pragma once
#include "BitArray.h"
#include "sliding_queue.h"

class VertexSubset {
public:
  bool all;
  bool is_sparse;
  uint64_t max_el;
  BitArray *ba = NULL;
  SlidingQueue<int32_t> *queue = NULL;
  QueueBuffer<int32_t> *queue_array = NULL;

  bool has(uint64_t i) {
    if (all) {
      return true;
    }
    if (is_sparse) {
      printf("shouldn't be calling has, is currently sparse\n");
      exit(-1);
      return false;
    } else {
      return ba->get(i);
    }
  }
  bool has_dense_no_all(uint64_t i) { return ba->get(i); }
  void has_dense_no_all_prefetch(uint64_t i) { return ba->prefetch(i); }

  uint64_t get_n() {
    // printf("get_n: is_sparse = %d, remove_duplicates = %d\n", is_sparse,
    // remove_duplicates);
    if (all) {
      return max_el;
    } else if (is_sparse) {
      return queue->size();
    } else {
      // printf("count = %lu\n", curr_ba->count());
      return ba->count();
    }
  }
  bool not_empty() {
    if (all) return true;
    else if (is_sparse) return queue->size() > 0;
    else return ba->not_empty();
  }
  void print() {
    printf("is_sparse = %d\n", is_sparse);
    if (all) {
      printf("{0,...,%lu}\n", max_el);
    } else if (is_sparse) {
      const uint32_t start = queue->shared_out_start;
      const uint32_t end = queue->shared_out_end;
      printf("{");
      for (uint32_t i = start; i < end; i++) {
        printf("%d, ", queue->shared[i]);
      }
      printf("}\n");
    } else {
      printf("{");
      for (uint32_t i = 0; i < max_el; i++) {
        if (ba->get(i)) {
          printf("%d, ", i);
        }
      }
      printf("}\n");
    }
  }
  void insert(uint64_t i) {
    if (is_sparse) {
      queue_array[4 * getWorkerNum()].push_back(i);
      return;
    } else {
      return ba->set(i);
    }
  }
  void insert_dense(uint64_t i) {
    return ba->set(i);
  }
  void insert_sparse(uint64_t i) {
    queue_array[4 * getWorkerNum()].push_back(i);
    return;
  }
  template <class F> void map(F &f) {
    if (all) {
      parallel_for(uint64_t i = 0; i < max_el; i++) { f.update(i); }
      return;
    }
    if (is_sparse) {
      const uint32_t start = queue->shared_out_start;
      const uint32_t end = queue->shared_out_end;
      parallel_for(uint32_t i = start; i < end; i++) {
        f.update(queue->shared[i]);
      }
    } else {
      return ba->map(f);
    }
  }
  template <class F> void map_sparse(F &f) {
    //printf("queue in map = %p\n", queue);
    const uint32_t start = queue->shared_out_start;
    const uint32_t end = queue->shared_out_end;
    parallel_for(uint32_t i = start; i < end; i++) {
      f.update(queue->shared[i]);
    }
  }
  // used to retunr empty vertexsubsets when we have no output
  VertexSubset() {}

  VertexSubset(el_t e, uint64_t max_el_, bool all_ = false)
      : all(all_), is_sparse(true), max_el(max_el_) {
    if (all) {
      is_sparse = false;
      return;
    }
    queue = new SlidingQueue<int32_t>(max_el);
    queue->push_back(e);
    queue->slide_window();
  }
  VertexSubset(bool *els, uint64_t len)
      : all(false), is_sparse(false), max_el(len) {
    ba = new BitArray(max_el);
    parallel_for_256(uint64_t i = 0; i < max_el; i++) {
      if (els[i]) {
        ba->set(i);
      }
    }
  }

  VertexSubset(const VertexSubset &other)
      : all(other.all), is_sparse(other.is_sparse), max_el(other.max_el), ba(other.ba), queue(other.queue) {
        //printf("queue = %p\n", queue);
  }
  VertexSubset& operator=(const VertexSubset& other) {
    all = other.all;
    is_sparse = other.is_sparse;
    max_el = other.max_el;
    ba = other.ba;
    queue = other.queue;
    queue_array = NULL;
    return *this;
  }

  // can't add anything to these once they have been copied,
  // just for keeping state like pushing past frontiers into a vector
  VertexSubset(const VertexSubset &other, bool copy_data)
      : all(other.all), is_sparse(other.is_sparse), max_el(other.max_el) {
    if (copy_data) {
      if (all) {
        return;
      }
      ba = NULL;
      queue = NULL;
      queue_array = NULL;
      if (is_sparse) {
        if (other.queue) {
          queue = new SlidingQueue<int32_t>(*other.queue, max_el);
        }
        if (other.queue_array) {
          queue_array = (QueueBuffer<int32_t> *)malloc(
              4 * sizeof(QueueBuffer<int32_t>) * getWorkers());
          for (int i = 0; i < getWorkers(); i++) {
            new (&queue_array[i * 4])
                QueueBuffer<int32_t>(*queue, other.queue_array[i * 4].local_size);
            queue_array[i * 4].in = other.queue_array[i * 4].in;
            memcpy(queue_array[i * 4].local_queue,
                   other.queue_array[i * 4].local_queue,
                   queue_array[i * 4].in * sizeof(int32_t));
          }
        }

      } else {
        if (other.ba) {
          ba = new BitArray(*other.ba);
        }
      }
    } else { // just create something similar where we will push the next set of data into
      // sparse and dense stay they way they are, will be changed by something else
      // all turns to dense, if we knew it was going to stay as all we would have no output and not use a new vertexsubset anyway
      if (is_sparse) {
        queue = new SlidingQueue<int32_t>(max_el);
        queue_array = (QueueBuffer<int32_t> *)malloc(
            4 * sizeof(QueueBuffer<int32_t>) * getWorkers());
        for (int i = 0; i < getWorkers(); i++) {
          new (&queue_array[i * 4]) QueueBuffer<int32_t>(*queue);
        }
      } else {
        all = false;
        ba = new BitArray(max_el);
      }
    }
  }
  void finalize() {
    if (is_sparse) {
      //queue->reset();
      parallel_for_1(int i = 0; i < getWorkers(); i++) {
        queue_array[i * 4].flush();
        queue_array[i * 4].~QueueBuffer();
      }
      queue->slide_window();
      free(queue_array);
    }
  }
  void del() {
    if (ba != NULL) {
      delete ba;
    }
    if (queue != NULL) {
      delete queue;
      //printf("deleteing queue %p\n", queue);
      queue = NULL;
    }
  }

  void convert_to_dense() {
    if (all || !is_sparse) {
      return;
    }
    // printf("converting sparse to dense\n");
    is_sparse = false;
    ba = new BitArray(max_el);
    // need an atomic setter to work in parallel
    for (uint32_t i = queue->shared_out_start; i < queue->shared_out_end; i++) {
      ba->set(queue->shared[i]);
    }
  }
  void convert_to_sparse() {
    if (all || is_sparse) {
      return;
    }
    // printf("converting dense to sparse\n");
    is_sparse = true;
    queue = new SlidingQueue<int32_t>(max_el);
    queue_array = (QueueBuffer<int32_t> *)malloc(
        4 * sizeof(QueueBuffer<int32_t>) * getWorkers());
    for (int i = 0; i < getWorkers(); i++) {
      new (&queue_array[i * 4]) QueueBuffer<int32_t>(*queue);
    }
    parallel_for(uint32_t i = 0; i < max_el; i++) {
      if (ba->get(i)) {
        queue_array[4 * getWorkerNum()].push_back(i);
      }
    }
    parallel_for(int i = 0; i < getWorkers(); i++) {
      queue_array[i * 4].flush();
    }
    queue->slide_window();
  }
};
