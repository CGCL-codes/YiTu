#ifndef HITREE_NODES_H
#define HITREE_NODES_H

#include "buckets.h"
#include "blocks.h"
#include "conflicts.h"
#include "linear_model.h"
#include "common.h"

#include <new>
#include <cmath>

#define BIT_TYPE uint8_t
#define BIT_SIZE (sizeof(BIT_TYPE) * 8)
#define BIT_LEN(x) (std::ceil((x) * 1. / BIT_SIZE))
#define BIT_GRAIN(x) (std::ceil((x) * 1. / GRAIN))
#define BIT_IDX(x) ((x) / BIT_SIZE)
#define BIT_POS(x) ((x) % BIT_SIZE)
#define SET_BIT_ONE(x, n) ((x) |= (1 << (n)))
#define SET_BIT_ZERO(x, n) ((x) &= (~(1 << (n))))
#define REV_BIT(x, n) ((x) ^= (1 << (n)))
#define GET_BIT(x, n) (((x) >> (n)) & 1)

namespace hitreespace {

template<typename KT, typename VT>
class TNode;

struct HyperParameter {
  // Parameters
  uint32_t max_bucket_size_ = 6;
  uint32_t aggregate_size_ = 0;
  // Constant parameters
  const uint32_t kMaxBucketSize = 6;
  const uint32_t kMinBucketSize = 1;
  const double kSizeAmplification = EXPAND_PARA;
  const double kSizeAmplificationDense = EXPAND_PARA;
  const double kTailPercent = 0.99;
};

enum EntryType {
  Unsued = 0,
  Edge = 1,
  Block = 2,
  HTNode = 3
};

template<typename KT, typename VT>
union Entry {
  Bucket<KT, VT>*     bucket_;      // The bucket pointer.
  TNode<KT, VT>*      child_;       // The child node pointer.
  std::pair<KT, VT>   kv_;

  Entry() { }
};

template<typename KT, typename VT>
class TNode {
typedef std::pair<KT, VT> KVT;
public:
  LinearModel<KT>*    model_;
  uint32_t            size_;
  uint32_t            capacity_;
  uint32_t            size_sub_tree_;
  uint32_t            index_num_;
  uint8_t*            bitmap0_;     // The i-th bit indicates whether the i-th 
                                    // position has a bucket or a child node.
  uint8_t*            bitmap1_;     // The i-th bit indicates whether the i-th 
                                    // position is a bucket.
  Entry<KT, VT>*      entries_;     // The pointer array that stores the pointer 
                                    // of buckets or child nodes.

public:
  // Constructor and deconstructor
  explicit TNode() : model_(nullptr), size_(0), capacity_(0), 
                      size_sub_tree_(0), bitmap0_(nullptr), 
                      bitmap1_(nullptr), entries_(nullptr) { }

  ~TNode() {
    destory_self();
  }

  // Get functions
  inline uint32_t size() const { return size_; }

  inline uint32_t capacity() const { return capacity_; }

  inline uint32_t size_sub_tree() const { return size_sub_tree_; }

  uint8_t entry_type(uint32_t idx) {
    uint32_t bit_idx = BIT_IDX(idx);
    uint32_t bit_pos = BIT_POS(idx);
    uint8_t bit0 = GET_BIT(bitmap0_[bit_idx], bit_pos);
    uint8_t bit1 = GET_BIT(bitmap1_[bit_idx], bit_pos);
    return ((bit1 << 1) | bit0);
  }

  uint64_t get_ima() {
   std::stack<TNode<KT, VT>*> stack;
   stack.push(this);
   uint64_t ret = 0;
   while (!stack.empty()) {
    TNode<KT, VT>* node = stack.top();
    stack.pop();
    ret++;
    if (node->model_ != nullptr) {
      for (auto i = 0; i < node->capacity_; i++) {
        auto type = node->entry_type(i);
        if (type == Edge) {
        } else if (type == Block) {
          i += (GRAIN / sizeof(KT) - 1);
        } else if (type == HTNode) {
          stack.push(node->entries_[i>>1].child_);
          i += (GRAIN / sizeof(KT) - 1);
        }
      }
    }
   }
   // printf("vetex %d\n", ret);
   return ret;
  }

  uint64_t get_size() {
   std::stack<TNode<KT, VT>*> stack;
   stack.push(this);
   uint64_t ret = 0;
   while (!stack.empty()) {
    TNode<KT, VT>* node = stack.top();
    stack.pop();
    if (node->model_ != nullptr) {
      ret += (sizeof(KT) + 2) * node->capacity_;
      ret += sizeof(TNode<KT, VT>);
      ret += sizeof(LinearModel<KT>);
      for (auto i = 0; i < node->capacity_; i++) {
        auto type = node->entry_type(i);
        if (type == Edge) {
        } else if (type == Block) {
          i += (GRAIN / sizeof(KT) - 1);
        } else if (type == HTNode) {
          stack.push(node->entries_[i>>1].child_);
          i += (GRAIN / sizeof(KT) - 1);
        }
      }
    }
   }
   // printf("vetex %d\n", ret);
   return ret;
  }

  template<class F> 
  void map_tree(F &f){
   uint32_t grain_kt = GRAIN / sizeof(KT);
   auto keys_ = reinterpret_cast<KT *>(entries_);    
   if(model_ != nullptr){
    TNode<KT, VT>* child_ptr = nullptr;
    for(auto i = 0 ; i < capacity_ ;i++){
      auto type = entry_type(i);
      if(type == Edge){
        f.update(keys_[i]);
      }else if(type == Block){
        auto bsize = keys_[i];
        for(auto j = 0; j < bsize; j++){
          f.update(keys_[i+j+1]);
        }
        i += (GRAIN / sizeof(KT) - 1);
      }else if(type == HTNode && entries_[i>>1].child_ != nullptr){
        if (child_ptr == entries_[i>>1].child_) {
          i += (GRAIN / sizeof(KT) - 1);
          continue;
        }
        child_ptr = entries_[i>>1].child_;
        entries_[i>>1].child_->map_tree(f);
        i += (GRAIN / sizeof(KT) - 1);
      }else{

      }
    }
   }else{
    if (index_num_ == 0) {
      for(auto i = 0 ; i < size_; i++){
        f.update(keys_[i]);
      }
    } else {
      uint32_t block_num = keys_[0];
      for (auto i = 0; i < block_num; ++i) {
        for (auto j = 0; j < keys_[grain_kt*(i+index_num_)]; j++) {
          f.update(keys_[grain_kt*(i+index_num_) + j + 1]);
        }
      }
    }
   }
  }

  template <class F>
  void map_tree_dense(F& f, bool &ret) {
    if (ret) return;
    uint32_t grain_kt = GRAIN / sizeof(KT);
    auto keys_ = reinterpret_cast<KT *>(entries_);    
    if(model_ != nullptr){
      TNode<KT, VT>* child_ptr = nullptr;
      for(auto i = 0 ; i < capacity_ ;i++){
        auto type = entry_type(i);
        if(type == Edge){
            f.update(keys_[i]);
            if (f.f.cond(f.self_index) == 0) {
              // printf("in \n");
              ret = true;
                return;
            }
        }else if(type == Block){
          auto bsize = keys_[i];
          for(auto j = 0; j < bsize; j++){
                f.update(keys_[i+j+1]);
                if (f.f.cond(f.self_index) == 0) {
              // printf("in \n");
                  ret = true;
                    return;
                }
          }
          i += (GRAIN / sizeof(KT) - 1);
        }else if(type == HTNode && entries_[i>>1].child_ != nullptr){
          if (child_ptr == entries_[i>>1].child_)  {
            i += (GRAIN / sizeof(KT) - 1);
            continue;
          }
          child_ptr = entries_[i>>1].child_;
          entries_[i>>1].child_->map_tree_dense(f, ret);
          if (ret) return;
          i += (GRAIN / sizeof(KT) - 1);
        }else{
          if (ret) return;
        }
      }
    } else {
      if (index_num_ == 0) {
        for(auto i = 0 ; i < size_; i++){
          f.update(keys_[i]);
          if (f.f.cond(f.self_index) == 0) {
            ret = true;
            return;
          }
        }
      } else {
        uint32_t block_num = keys_[0];
        for (auto i = 0; i < block_num; ++i) {
          for (auto j = 0; j < keys_[grain_kt*(i+index_num_)]; j++) {
            f.update(keys_[grain_kt*(i+index_num_) + j + 1]);
            if (f.f.cond(f.self_index) == 0) {
              ret = true;
              return;
            }
          }
        }
      }
   }
  }

  // this is not traverse by DFS
  template <class F>
  void map_tree_dense(F& f) {
   std::stack<TNode<KT, VT>*> stack;
   stack.push(this);
   while (!stack.empty()) {
    TNode<KT, VT>* node = stack.top();
    auto keys_ = reinterpret_cast<KT *>(node->entries_);    
    stack.pop();
    if (node->model_ != nullptr) {
      for (auto i = 0; i < node->capacity_; i++) {
        // TNode<KT, VT>* child_ptr;
        auto type = node->entry_type(i);
        if (type == Edge) {
          f.update(keys_[i]);
          if (f.f.cond(f.self_index) == 0) {
              return;
          }
        } else if (type == Block) {
          auto bsize = keys_[i];
          for (auto j = 0; j < bsize; j++) {
              f.update(keys_[i+j+1]);
              if (f.f.cond(f.self_index) == 0) {
                  return;
              }
          }
          i += (GRAIN / sizeof(KT) - 1);
        } else if (type == HTNode) {
          // if (child_ptr == node->entries_[i].child_) continue;
          // child_ptr = node->entries_[i].child_;
          // stack.push(child_ptr);
          stack.push(node->entries_[i>>1].child_);
          i += (GRAIN / sizeof(KT) - 1);
        }
      }
    } else {
      for (auto i = 0; i < node->size_; i++) {
        f.update(keys_[i]);
        if (f.f.cond(f.self_index) == 0) {
          return;
        }
      }
    }
   }
  }

  class TNodeIterator {
  private:
   TNode<KT, VT>* node_;
   uint32_t idx_;
   std::stack<TNode<KT, VT>*> stack;
  public:
   TNodeIterator(TNode<KT, VT>* node) : node_(node), idx_(0) {}

   bool hasNext() {
    if (idx_ < node_->size_) {
      return true;
    } else if (node_->model_ != nullptr && node_->size_ != 0) {
      for (auto i = idx_; i < node_->capacity_; i++) {
        auto type = node_->entry_type(i);
        if (type == Edge) {
          return true;
        } else if (type == Block) {
          return true;
        } else if (type == HTNode) {
          TNode<KT, VT>* child = node_->entries_[i].child_;
          stack.push(child);
        }
      }
      while (!stack.empty()) {
        TNode<KT, VT>* node = stack.top();
        stack.pop();
        if (node->model_ != nullptr && node->size_ != 0) {
          for (auto i = 0; i < node->capacity_; i++) {
            auto type = node->entry_type(i);
            if (type == Edge) {
              return true;
            } else if (type == Block) {
              return true;
            } else if (type == HTNode) {
              TNode<KT, VT>* child = node->entries_[i].child_;
              stack.push(child);
            }
          }
        } else {
          return true;
        }
      }
    }
    return false;
   }

   TNodeIterator operator++(void) {
    if (idx_ < node_->size_) {
      idx_++;
    } else if (node_->model_ != nullptr && node_->size_ != 0) {
      for (auto i = idx_; i < node_->capacity_; i++) {
        auto type = node_->entry_type(i);
        if (type == Edge) {
          idx_ = i + 1;
          return *this;
        } else if (type == Block) {
          idx_ = i + 1;
          return *this;
        } else if (type == HTNode) {
          TNode<KT, VT>* child = node_->entries_[i].child_;
          stack.push(child);
        }
      }
      while (!stack.empty()) {
        TNode<KT, VT>* node = stack.top();
        stack.pop();
        if (node->model_ != nullptr && node->size_ != 0) {
          for (auto i = 0; i < node->capacity_; i++) {
            auto type = node->entry_type(i);
            if (type == Edge) {
              idx_ = i + 1;
              return *this;
            } else if (type == Block) {
              idx_ = i + 1;
              return *this;
            } else if (type == HTNode) {
              TNode<KT, VT>* child = node->entries_[i].child_;
              stack.push(child);
            }
          }
        } else {
          idx_ = 0;
          node_ = node;
          return *this;
        }
      }
    }
    return *this;
   }

  TNode<KT, VT>* operator*() {
    return node_;
   }

  uint32_t getIdx(){
    return idx_;
  }

  TNodeIterator end() { return TNodeIterator(nullptr); }
  };

 private:
  void set_entry_type(uint32_t idx, uint8_t type) {
    uint32_t bit_idx = BIT_IDX(idx);
    uint32_t bit_pos = BIT_POS(idx);
    if (GET_BIT(bitmap0_[bit_idx], bit_pos) ^ GET_BIT(type, 0)) {
      REV_BIT(bitmap0_[bit_idx], bit_pos);
    }
    if (GET_BIT(bitmap1_[bit_idx], bit_pos) ^ GET_BIT(type, 1)) {
      REV_BIT(bitmap1_[bit_idx], bit_pos);
    }
  }

  void set_entry_grain(uint32_t idx, uint8_t type, uint32_t size) {
    uint32_t bit_idx = BIT_IDX(idx);
    for (uint32_t i = 0; i < (size / BIT_SIZE); ++i) {
      bitmap1_[bit_idx+i] = 0xff;
      if (GET_BIT(type, 0)) {
        bitmap0_[bit_idx+i] = 0xff;
      } else {
        bitmap0_[bit_idx+i] = 0x00;
      }
    }
  }

  uint32_t get_grain_ones(uint32_t idx, uint32_t size) {
    uint32_t ones = 0;
    uint32_t bit_idx = BIT_IDX(idx);
    for (uint32_t i = 0; i < size / BIT_SIZE; ++i) {
      uint8_t t = bitmap0_[bit_idx+i];
      while (t) {
        ones += (t & 0x01);
        t >>= 1;
      }
    }
    return ones;
  }

public:
  TNodeIterator begin() { return TNodeIterator(this); }
  // User API interfaces
  bool find(KT key) {
    KT* keys_ = reinterpret_cast<KT *>(entries_);
    const uint32_t grain_kt = GRAIN / sizeof(KT);
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(key), 0L), 
                              static_cast<int64_t>(capacity_ - 1));
      uint8_t type = entry_type(idx);
      // if (key == 98302)
      //   std::cout << "idx:" << idx << " "
      //             << "type:" << type + 0 << std::endl;
      if (type == Edge && compare(keys_[idx], key)) {
        return true;
      } else if (type == Block) {
        return block_find(&(keys_[PALIGN_DOWN(idx, grain_kt)]), key);
      } else if (type == HTNode) {
        return entries_[PALIGN_DOWN(idx, grain_kt) >> 1].child_->find(key);
      } else {
        return false;
      }
    } else {
      if (index_num_ == 0) {
        for(auto i = 0 ; i < size_; i++){
          if (compare(keys_[i], key)) {
            return true;
          }
        }
      } else {
        // uint32_t block_num = keys_[0];
        // for (uint32_t i = 0; i < block_num; ++i) {
        //   for (uint32_t j = 0; j < keys_[grain_kt*(i+index_num_)]; j++) {
        //     if (compare(keys_[grain_kt*(i+index_num_) + j + 1], key)) {
        //       return true;
        //     }
        //   }
        // }
        // print_hitree();
        // exit(0);
        uint32_t block_num = keys_[0];
        uint32_t block_id = 0;
        bool get_bid = false;
        for (int32_t i = 0; i < block_num; ++i) {
          if (keys_[1+i] > key) {
            block_id = std::max(i-1, (int32_t)0);
            get_bid = true;
            break;
          }
        }
        if (get_bid == false) block_id = block_num-1;
        for (uint32_t j = 0; j < keys_[grain_kt*(block_id+index_num_)]; j++) {
          if (compare(keys_[grain_kt*(block_id+index_num_) + j + 1], key)) {
            return true;
          }
        }
      }
      return false;
    }
  }

  void visit_edges_test(std::vector<uint32_t> &array){
    if (model_ != nullptr) {
      for (auto i = 0; i < capacity_; i++) {
        auto type = entry_type(i);
        TNode<KT, VT>* child_ptr;
        if (type == Edge) {
          auto d = entries_[i].kv_.first;
          array.push_back(d);
        } else if (type == Block) {
          auto bsize = entries_[i].bucket_->size_;
            for (auto j = 0; j < bsize; j++) {
              auto d = entries_[i].bucket_->data_[j].first;
              array.push_back(d);
            }
        } else if (type == HTNode && entries_[i].child_ != nullptr) {
          if (child_ptr == entries_[i].child_)
            continue;
          child_ptr = entries_[i].child_;
          entries_[i].child_->visit_edges_test(array);
        } else {
            continue;
        }
      }
    } else {
        for (auto i = 0; i < size_; i++) {
          auto d = entries_[i].kv_.first;
          array.push_back(d);
      }
    }
  }

  void visit_edges(){
    if (model_ != nullptr){
      for (auto i = 0; i < capacity_; i++){
       auto type = entry_type(i);
       TNode<KT,VT>* child_ptr;
       if (type == Edge){
        auto d = entries_[i].kv_.first;
        //std::cout << d << " "; 
       }else if (type == Block){
        auto bsize = entries_[i].bucket_->size_;
        for (auto j = 0; j < bsize; j++){
          auto d = entries_[i].bucket_->data_[j].first;
          //std::cout << d << " ";
        }
       }else if (type == HTNode && entries_[i].child_ != nullptr){
        if (child_ptr == entries_[i].child_)
          continue;
        child_ptr = entries_[i].child_;
        entries_[i].child_->visit_edges();
       }else{
        continue;
       }
      }
    }else{
      for (auto i = 0; i < size_; i++){
        auto d = entries_[i].kv_.first;
        //std::cout << d << " ";
      }
    }
  }

  KT get_first_key() {
    auto keys_ = reinterpret_cast<KT*>(entries_);
    auto grain_kt = GRAIN / sizeof(KT);
    if(model_ != nullptr){
      for (auto i = 0; i < capacity_; i++){
        auto type = entry_type(i);
        if (type == Edge){
          return keys_[i];
        }else if (type == Block){
          auto bsize = keys_[i];
          for (auto j = 0; j < bsize; j++){
            return keys_[i+j+1];
          }
          i += (GRAIN / sizeof(KT) - 1);
        }else if(type == HTNode){
          return entries_[i >> 1].child_->get_first_key();
          i += (GRAIN / sizeof(KT) - 1);
        }
      }
    }else{
      if (index_num_ == 0) {
        for (auto i = 0; i < size_; i++) {
          return keys_[i];
        }
      } else {
        uint32_t block_num = keys_[0];
        for (auto i = 0; i < block_num; ++i) {
          for (auto j = 0; j < keys_[grain_kt * (i + index_num_)]; j++) {
            return keys_[grain_kt * (i + index_num_) + j + 1];
          }
        }
      }
     }
  }

  void print_hitree(std::vector<uint32_t>&nei, int level = 0) {
    auto keys_ = reinterpret_cast<KT *>(entries_);
    auto grain_kt = GRAIN / sizeof(KT);
    if (model_ != nullptr) {
      TNode<KT, VT>* child_ptr = nullptr;
      for (auto i = 0; i < capacity_; ++i) {
        auto type = entry_type(i);
        if (type == Edge) {
          nei.push_back(keys_[i]);
        } else if (type == Block) {
          auto bsize = keys_[i];
          for (auto j = 0; j < bsize; ++j) {
            nei.push_back(keys_[i+j+1]);
          }
          i += (GRAIN / sizeof(KT) - 1);
        } else if (type == HTNode && entries_[i>>1].child_ != nullptr) {
          if (child_ptr == entries_[i>>1].child_) {
          i += (GRAIN / sizeof(KT) - 1);
          continue;
        }
        child_ptr = entries_[i>>1].child_;
         entries_[i>>1].child_->print_hitree(nei, level + 1);
          i += (GRAIN / sizeof(KT) - 1);
        } else {
          continue;
        }
      }
    } else {
      if (index_num_ == 0) {
        for(auto i = 0 ; i < size_; i++){
          nei.push_back(keys_[i]);
        }
      } else {
        uint32_t block_num = keys_[0];
        for (auto i = 0; i < block_num; ++i) {
          for (auto j = 0; j < keys_[grain_kt*(i+index_num_)]; j++) {
            nei.push_back(keys_[grain_kt*(i+index_num_) + j + 1]);
          }
        }
      }
    }
  }


  bool update(KVT kv) {
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(kv.first), 0L), 
                              static_cast<int64_t>(size_ - 1));
      uint8_t type = entry_type(idx);
      if (type == Edge && compare(entries_[idx].kv_.first, kv.first)) {
        entries_[idx].kv_ = kv;
        return true;
      } else if (type == Block) {
        return entries_[idx].bucket_->update(kv);
      } else if (type == HTNode) {
        return entries_[idx].child_->update(kv);
      } else {
        return false;
      }
    } else {
      uint32_t idx = std::lower_bound(entries_, entries_ + size_, kv.first, 
                      [](const Entry<KT, VT>& kk, const KT k) {
                        return kk.kv_.first < k;
                      }) - entries_;
      if (idx < size_ && compare(entries_[idx].kv_.first, kv.first)) {
        entries_[idx].kv_ = kv;
        return true;
      } else {
        return false;
      }
    }
  }

  uint32_t remove(KT key) {
    KT *keys_ = reinterpret_cast<KT *>(entries_);
    const uint32_t  grain_kt = GRAIN / sizeof(KT);
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(key), 0L), 
                              static_cast<int64_t>(capacity_ - 1));
      uint8_t type = entry_type(idx);
      // if(key == 98302) std::cout<<"idx:"<<idx<<" "<<"type:"<<type+0<<std::endl;
      if (type == Edge && compare(keys_[idx], key)) {
        set_entry_type(idx, Unsued);
        size_ --;
        size_sub_tree_ --;
        return 1;
      } else if (type == Block) {
        uint32_t base = PALIGN_DOWN(idx, grain_kt);
        uint32_t res = block_remove(keys_ + base, key);
        size_sub_tree_ -= res;
        if(keys_[0] == 0){
          set_entry_grain(base, Unsued, grain_kt);
        }
        return res;
      } else if (type == HTNode) {
        uint32_t res =
            entries_[PALIGN_DOWN(idx, grain_kt) >> 1].child_->remove(key);
        //size == 0 set all Unsued
        if (res == -1){
          set_entry_grain(PALIGN_DOWN(idx, grain_kt), Unsued, grain_kt);
          res = 1;
        }
        size_sub_tree_ -= res;
        return res;
      } else {
        return 0;
      }
    } else {
      if (index_num_ == 0){
        uint32_t idx = std::lower_bound(keys_, keys_+ size_, key, 
                      [](const KT& kk, const KT key) {
                        return kk < key;
                      }) - keys_;
        // if(key == 2930360) std::cout<<"idx:"<<idx<<"keys_[idx]:"<<keys_[idx]<<std::endl;
        if (idx < size_ && compare(keys_[idx], key)) {
          if (size_ > 1) {
            for (uint32_t i = idx; i < size_ - 1; ++ i) {
              // if (keys_[i] != key) {
              //   break;
              // }
              keys_[i] = keys_[i+1];
            }
          }
          size_ --;
          size_sub_tree_ --;
          return 1;
        } else {
          return 0;
        }
      } else {
        //block remove
        uint32_t block_num = keys_[0];
        uint32_t block_id = 0;
        bool get_bid = false;
        for (int32_t i = 0; i < block_num; ++i) {
          if (keys_[i+1] > key) {
            block_id = std::max(i-1, 0);
            get_bid = true;
            break;
          }
        }
        if(get_bid == false){
          block_id = block_num - 1;
        }
        

        uint32_t block_addr = (index_num_ + block_id) * grain_kt;
        uint32_t data_addr = block_addr + 1;
        uint32_t block_size = keys_[block_addr];
        // if (key == 98302) {
        //   std::cout << "remove:" << key << std::endl;
        //   std::cout << "block_id:" << block_id << std::endl;
        //   std::cout << "block_num:" << block_num << std::endl;
        //   std::cout << "block_size:" << block_size << std::endl;
        //   std::cout << "keys_[data_addr + 0]:" << keys_[data_addr + 0] << std::endl;
        // }
        bool copy = false;
        bool is_first = false;
        for (uint32_t i = 0; i < block_size; ++i) {
          if (keys_[data_addr + i] == key) {
            if(i == 0) is_first = true;
            copy = true;
          }
          if (copy && i + 1 < block_size) {
            keys_[data_addr + i] = keys_[data_addr + i + 1];
          }
        }
        if(copy){
          // update index
          if(is_first) keys_[block_id + 1] = key; 
          // update block info
          size_--;
          keys_[block_addr] --;
          //rebuild
          if(keys_[block_addr] == 0){
            uint32_t bdis = 1;
            while ((int32_t)(block_id - bdis) >= (int32_t)(0) || block_id + bdis < block_num ){
              //rebuild with left
              if ((int32_t)(block_id - bdis) >= (int32_t)(0) && 
                  (bdis+1) * grain_kt / (bdis*grain_kt + keys_[block_addr - bdis * grain_kt]) > EXPAND_PARA ) {
                if (bdis > log(1.0 * capacity_ / grain_kt - index_num_)) break;
                rebuild_blocks(block_id - bdis, block_id);
                return 1;
              } 
              //rebuild with right
              if (block_id + bdis < block_num && 
                  (bdis+1) * grain_kt / (bdis*grain_kt + keys_[block_addr + bdis * grain_kt]) > EXPAND_PARA ) {
                if (bdis > log(1.0 * capacity_ / grain_kt - index_num_)) break;
                rebuild_blocks(block_id, block_id + bdis);
                return 1;
              }
              bdis++;
            }
          }

          // child size = 0
          if (size_ == 0) {
            return -1;
          }

          return 1;
        }else{
          return 0;
        }



      }
    }
  }

  uint32_t merge(KT *ks, uint32_t begin, uint32_t end, KT k, bool block = false) {
    // print_hitree();
    uint32_t size = 0;
    bool ik = false;
    KT* keys_ = reinterpret_cast<KT *>(entries_);
    for (uint32_t i = begin; i < end; ++i) {
      if (block || (entry_type(i) == Edge)) {
        if (ik || keys_[i] < k) {
          ks[size++] = keys_[i];
        } else {
          ks[size++] = k;
          ks[size++] = keys_[i];
          ik = true;
        }
      }
    }
    if (!ik) ks[size++] = k;

    // for (int i = 0; i < size; ++i) {
    //   if (ks[i] == 0) {
    //     printf("error, ks data is zero\n");
    //     printf("size k block = %d %d %d\n", k, size, block);
    //     for (uint32_t j = begin; j < end; ++j) {
    //       if (block || (entry_type(j) == Edge)) {
    //         printf("%d ", keys_[j]);
    //       }
    //     }
    //     printf("\n");
    //     for (int z = 0; z < size; ++z) {
    //       printf("%d ", ks[z]);
    //     }
    //     printf("\n");
    // print_hitree();
    //     exit(0);
    //   }
    // }
    return size;
  }

  void rebuild_blocks(uint32_t bs, uint32_t be){
    const uint32_t grain_kt = GRAIN / sizeof(KT);
    KT* keys_ = reinterpret_cast<KT *>(entries_);
    uint32_t size = 0;
    KT *ks = new KT[grain_kt*(be - bs + 1)];
    //copy to an array
    for (uint32_t b = bs; b <= be; ++b) {
      uint32_t block_addr = (index_num_+b) * grain_kt;
      uint32_t bsize = keys_[block_addr];
      for (uint32_t i = 0; i < bsize; ++i) {
        ks[size++] = keys_[block_addr+1+i];
      }
    }
    //rebuild
    uint32_t each_size = size / (be - bs + 1);
    uint32_t ks_idx = 0;
    for (uint32_t b = bs; b <= be; ++b) {
        uint32_t block_addr = (index_num_ + b) * grain_kt;
        keys_[block_addr] =
            each_size + (b < (size % (be - bs + 1) + bs) ? 1 : 0);
        keys_[1 + b] = ks[ks_idx];
        for (uint32_t i = 0; i < keys_[block_addr]; ++i) {
            keys_[block_addr + 1 + i] = ks[ks_idx++];
        }
    }
    delete[] ks;
    size_++;
  }



  // rebuild blocks of [bs, be]
  void rebuild_blocks(uint32_t bs, uint32_t be, uint32_t k) {
    // printf("%d %d %d\n", bs, be, k);
    // merge to an array
    const uint32_t grain_kt = GRAIN / sizeof(KT);
    KT* keys_ = reinterpret_cast<KT *>(entries_);
    uint32_t size = 0;
    KT *ks = new KT[grain_kt*(be - bs + 1)];
    bool ink = false;
    for (uint32_t b = bs; b <= be; ++b) {
      uint32_t block_addr = (index_num_+b) * grain_kt;
      uint32_t bsize = keys_[block_addr];
      for (uint32_t i = 0; i < bsize; ++i) {
        if (ink == false && keys_[block_addr + 1 + i] > k) {
          ks[size++] = k;
          ink = true;
        }
        ks[size++] = keys_[block_addr+1+i];
      }
    }
    if (ink == false) ks[size++] = k;
            // for (int i = 1; i < size; ++i) {
            //   if (ks[i] < ks[i-1]) {
            //     printf("build data not sort\n");
            //     for (int j = 0; j < size; ++j) {
            //       printf("%d ", ks[j]);
            //     }
            //     printf("\n");
            //     exit(0);
            //   }
            // }
    // redistribute to blocks
    uint32_t each_size = size / (be-bs+1);
    uint32_t ks_idx = 0;
    // uint32_t cur_size = 0;
    for (uint32_t b = bs; b <= be; ++b) {
      uint32_t block_addr = (index_num_+b) * grain_kt;
      keys_[block_addr] = each_size + (b < (size % (be-bs+1) + bs) ? 1 : 0);
      keys_[1+b] = ks[ks_idx];
      // if (keys_[block_addr] >= grain_kt || keys_[block_addr] == 0) {
      //   printf("%d %d \n", size, b);
      //   printf("biuld block error\n");
      //   exit(0);
      // }
      for (uint32_t i = 0; i < keys_[block_addr]; ++i) {
        keys_[block_addr+1+i] = ks[ks_idx++];
      }
    }
    delete [] ks;
    size_++;
  }

  void insert(KT k, uint32_t depth, const HyperParameter& hyper_para) {
    size_sub_tree_ ++;
    KT* keys_ = reinterpret_cast<KT *>(entries_);
    const uint32_t grain_kt = GRAIN / sizeof(KT);
    if (model_ != nullptr) {
      uint32_t idx = std::min(std::max(model_->predict(k), 0L), 
                              static_cast<int64_t>(capacity_ - 1));
      uint8_t type = entry_type(idx);
      if (type == Unsued) {
        set_entry_type(idx, Edge);
        keys_[idx] = k;
        size_++;
        // if (k == 0) {
        //   printf("insert zero, idx = %d\n", idx);
        // }
      } else if (type == HTNode) {
        entries_[PALIGN_DOWN(idx, grain_kt) >> 1].child_->insert(k, depth + 1, hyper_para);
      } else if (type == Edge) {
        if (keys_[idx] == k) return;
        uint32_t base = PALIGN_DOWN(idx, grain_kt);
        // auto size = get_grain_ones(base, grain_kt);
        KT ks[grain_kt*2];
        // printf("DATA %d\n", depth);
        uint32_t size = merge(ks, base, base + grain_kt, k);
        if (size < (grain_kt-1)) {
          set_entry_grain(base, Block, grain_kt);
          make_block(keys_+base, ks, size);
          size_++;
        } else {
          set_entry_grain(base, HTNode, grain_kt);
          entries_[base >> 1].child_ = new TNode<KT, VT>();
          entries_[base >> 1].child_->build(ks, size, depth + 1, hyper_para);
        }
      } else if (type == Block) {
        uint32_t base = PALIGN_DOWN(idx, grain_kt);
        if (keys_[base] < (grain_kt-1)) {
          set_entry_grain(base, Block, grain_kt);
          block_insert_sort(keys_+base, k);
          size_++;
        } else {
          KT ks[grain_kt];
          // printf("BUCKET %d %d\n", depth, keys_[base]);
          uint32_t size = merge(ks, base+1, base+1+keys_[base], k, true);
          // printf("BUCKET %d %d %d\n", depth, keys_[base], size);
          set_entry_grain(base, HTNode, grain_kt);
          entries_[base >> 1].child_ = new TNode<KT, VT>();
          entries_[base >> 1].child_->build(ks, size, depth + 1, hyper_para);
        }
      }
    } else {
      // find
      if (index_num_ == 0) {
        if (size_ < capacity_) {
          int32_t idx = std::lower_bound(keys_, keys_ + size_, k,
                      [](const KT& kk, KT k) {
                        return kk < k;
                      }) - keys_;
          
          for(int32_t i = idx; i < size_; i++){
            if (compare(keys_[i], k))
              return;
          }
          for (int32_t i = size_; i > idx; --i) {
            keys_[i] = keys_[i - 1];
          }
          keys_[idx] = k;
          size_++;
        } else {
          // Copy data for rebuilding
          uint32_t node_size = size_;
          KT* ks = new KT[node_size + 1];
          for (uint32_t i = 0; i < size_; ++ i) {
            ks[i] = keys_[i];
          }
          ks[node_size] = k;
          std::sort(ks, ks + node_size + 1,
            [](auto const& a, auto const& b) {
              return a < b;
            });
          // Clear entry
          destory_self();
          // Create child node
          build(ks, node_size + 1, depth, hyper_para);
          delete[] ks;
        }
      } else {
        // find
        uint32_t block_num = keys_[0];
        uint32_t block_id = 0;
        bool get_bid = false;
        for (int32_t i = 0; i < block_num; ++i) {
          if (keys_[1+i] > k) {
            block_id = std::max(i-1, (int32_t)0);
            get_bid = true;
            break;
          }
        }
        if (get_bid == false) block_id = block_num-1;
        // insert
        uint32_t block_addr = (index_num_+block_id) * grain_kt;
        uint32_t data_addr  = block_addr+1;
        uint32_t block_size = keys_[block_addr];
        for(int32_t i = 0; i < block_size; i++){
          if(compare(keys_[data_addr + i], k)) return ;
        }
        if (block_size < grain_kt-1) { // insert to current block
          bool ink = false;
          for (int32_t i = block_size-1; i >= 0; --i) {
            keys_[data_addr+i+1] = keys_[data_addr+i];
            if (keys_[data_addr+i] < k) {
              keys_[data_addr+i+1] = k;
              ink = true;
              break;
            }
          }
          if (ink == false) {
            keys_[data_addr] = k;
            keys_[1+block_id] = k;
          }
          size_++;
          keys_[block_addr]++;
              // if (i == 0) keys_[1+block_id] = k;
            // for (int i = 1; i < block_size+1; ++i) {
            //   if (keys_[data_addr  + i] < keys_[data_addr+ i-1]) {
            //     printf("insert data not sort\n");
            //     for (int j = 0; j < block_size+1; ++j) {
            //       printf("%d ", keys_[data_addr + j ]);
            //     }
            //     printf("\n");
            //     exit(0);
            //   }
            // }
        } else {
          uint32_t bdis = 1;
          while ((int32_t)(block_id - bdis) >= (int32_t)(0) || block_id + bdis < block_num) {
            // insert to left
            if ((int32_t)(block_id - bdis) >= (int32_t)(0) && keys_[block_addr - bdis * grain_kt] < grain_kt - 1) {
            // if ((bdis+1) * grain_kt / (bdis*grain_kt + keys_[block_addr - bdis * grain_kt]) < 1.1) break;
            // if ((bdis+1) * grain_kt / (bdis*grain_kt + keys_[block_addr - bdis * grain_kt]) < hyper_para.kSizeAmplification) break;
            // if (bdis > (capacity_ / grain_kt - index_num_) / 2) break;
            // 1.the time complexity must less than pma; 2. avoid whole data's rebalance
              // if (bdis > log(1.0 * capacity_ / grain_kt - index_num_)) break;
              rebuild_blocks(block_id - bdis, block_id, k);
              return;
            }
            // insert to right
            if (block_id + bdis < block_num && keys_[block_addr + bdis * grain_kt] < grain_kt - 1) {
            // if ((bdis+1) * grain_kt / (bdis*grain_kt + keys_[block_addr + bdis * grain_kt]) < 1.1) break;
            // if ((bdis+1) * grain_kt / (bdis*grain_kt + keys_[block_addr + bdis * grain_kt]) < hyper_para.kSizeAmplification) break;
            // if (bdis > (capacity_ / grain_kt - index_num_) / 2) break;
              // if (bdis > log(1.0 * capacity_ / grain_kt - index_num_)) break;
              rebuild_blocks(block_id, block_id + bdis, k);
              return;
            }
            // if (bdis > 4) break;
            ++bdis;
          }
          uint32_t size = 0;
          KT *ks = new KT[size_+1];
          bool ink = false;
          for (uint32_t b = 0; b < block_num; ++b) {
            uint32_t block_addr = (index_num_+b) * grain_kt;
            uint32_t bsize = keys_[block_addr];
            for (uint32_t i = 0; i < bsize; ++i) {
              if (ink == false && keys_[block_addr + 1 + i] > k) {
                ks[size++] = k;
                ink = true;
              }
              ks[size++] = keys_[block_addr+1+i];
            }
          }
          if (ink == false) ks[size++] = k;
          // destory_self();
          // build_dense_node(ks, size, 1, size * hyper_para.kSizeAmplificationDense);
          if (size < DENSE) {
            // for (int i = 1; i < size; ++i) {
            //   if (ks[i] < ks[i-1]) {
            //     printf("build dense data not sort\n");
            //     for (int j = 0; j < size; ++j) {
            //       printf("%d ", ks[j]);
            //     }
            //     printf("\n");
            //     exit(0);
            //   }
            // }
            destory_self();
            build_dense_node(ks, size, 1, size * hyper_para.kSizeAmplificationDense);
            // build_dense_node(ks, size, 1, size * hyper_para.kSizeAmplification);
          } else {
            // printf("%d %d\n", size_+1, size);
            // for (int i = 0; i < size; ++i) {
            //   printf("%d ", ks[i]);
            // }
            // printf("\n");
            // for (int i = 1; i < size; ++i) {
            //   if (ks[i] < ks[i-1]) {
            //     printf("data not sort\n");
            //     exit(0);
            //   }
            // }
            destory_self();
            build(ks, size, 1, hyper_para);
          }
          delete [] ks;
        }
      }
    }
  }

  void destory_self() {    
    if (model_ != nullptr) {
      delete model_;
      model_ = nullptr;
      for (uint32_t i = 0; i < capacity_; ++ i) {
        uint8_t type_i = entry_type(i);
        if (type_i == Block) {
          i += ((GRAIN / sizeof(KT))-1);
        } else if (type_i == HTNode) {
          if (entries_[i>>1].child_ != nullptr) {
            delete entries_[i >> 1].child_;
            entries_[i>>1].child_ = nullptr;
          }
          i += ((GRAIN / sizeof(KT))-1);
        }
      }
      delete[] bitmap0_;
      bitmap0_ = nullptr;
      delete[] bitmap1_;
      bitmap1_ = nullptr;
    }
    if (entries_ != nullptr) {
      // delete[] entries_;
      ::operator delete(entries_, std::align_val_t{GRAIN});
      entries_ = nullptr;
    }
    size_ = 0;
    capacity_ = 0;
    size_sub_tree_ = 0;
  }

  void build_dense_node(const KT* ks, uint32_t size, uint32_t depth, 
                        uint32_t capacity) {
    model_ = nullptr;
    size_ = size;
    capacity_ = capacity;
    size_sub_tree_ = size;
    uint32_t grain_kt = GRAIN / sizeof(KT);
    if (size < grain_kt * 2) {
      index_num_ = 0;
      capacity_ = PALIGN_UP(capacity_, grain_kt);
      entries_ = new(std::align_val_t{GRAIN}) Entry<KT, VT>[(1+capacity_)/2];
      KT* keys_ = reinterpret_cast<KT *>(entries_);
      for (uint32_t i = 0; i < size; ++ i) {
        keys_[i] = ks[i];
      }
    } else { // use index
      uint32_t block_num = PALIGN_UP(capacity_, grain_kt) / grain_kt;
      uint32_t index_num = PALIGN_UP(block_num + 1, grain_kt) / grain_kt;
      uint32_t each_size = size / block_num;
      uint32_t each_rem  = size % block_num;
      uint32_t offset = index_num * grain_kt;
      // printf("%d %d %d %d\n", size, block_num, index_num, each_size);
      index_num_ = index_num;
      capacity_ = PALIGN_UP(capacity_, grain_kt) + index_num * grain_kt;
      entries_ = new(std::align_val_t{GRAIN}) Entry<KT, VT>[(1+capacity_)/2];
      KT* keys_ = reinterpret_cast<KT *>(entries_);
      keys_[0] = block_num;
      uint32_t ks_idx = 0;
      uint32_t cur_size = 0;
      for (uint32_t i = 0; i < block_num; ++i) {
        keys_[1+i] = ks[ks_idx];
        cur_size = each_size + (i < (size % block_num) ? 1 : 0);
        keys_[grain_kt*i + offset] = cur_size;
      // if (cur_size >= grain_kt || cur_size == 0) {
      //   printf("build dense error\n");
      //   exit(0);
      // }
      //   if ((int32_t)keys_[grain_kt*i + offset] < 0) {
      //     printf("size < 0\n");
      //     printf("%d %d %d %d %d %d\n", i, keys_[grain_kt*i + offset], size, block_num, index_num, each_size);
      //     exit(0);
      //   }
        for (uint32_t j = 0; j < cur_size; ++j) {
          keys_[offset + grain_kt*i + j + 1] = ks[ks_idx + j];
        }
        ks_idx += cur_size;
      }
    }
  }

  void build(const KT* ks, uint32_t size, uint32_t depth, 
              const HyperParameter& hyper_para) {
    // if ((size < 1024 && depth == 1) || size < GRAIN) {
    //   build_dense_node(ks, size, depth, size + hyper_para.max_bucket_size_);
    //   return;
    // }
    // if (size < GRAIN / sizeof(KT) * 2) {
    // if ((size < DENSE && depth == 1) || size < GRAIN*2) {
    // if ((size < DENSE && depth == 1) || size < GRAIN / 2) {
    if ((size < DENSE)) {
      build_dense_node(ks, size, depth, size * hyper_para.kSizeAmplificationDense);
      // build_dense_node(ks, size, depth, size * 2);
      return;
    }
    ConflictsInfo* ci = build_linear_model<KT, VT>(ks, size, model_, hyper_para.kSizeAmplification);
    if (ci == nullptr) {
      // printf("%d %d\n", size, depth);
      build_dense_node(ks, size, depth, size + hyper_para.max_bucket_size_);
    } else {
      index_num_ = 0;
      const uint32_t grain_kt = GRAIN / sizeof(KT);
      // Allocate memory for the node
      uint32_t bit_len = PALIGN_DOWN(static_cast<uint32_t>(BIT_LEN(ci->max_size_)), grain_kt / BIT_SIZE) + (grain_kt / BIT_SIZE);
      // capacity_ = PALIGN_UP(ci->max_size_, grain_kt);
      capacity_ = ci->max_size_;
      size_ = 0;
      size_sub_tree_ = size;
      bitmap0_ = new BIT_TYPE[bit_len];
      bitmap1_ = new BIT_TYPE[bit_len];
      entries_ = new(std::align_val_t{GRAIN}) Entry<KT, VT>[(ci->max_size_+1)/2];
      // entries_ = new Entry<KT, VT>[ci->max_size_];
      KT* keys_ = reinterpret_cast<KT *>(entries_);
      memset(bitmap0_, 0, sizeof(BIT_TYPE) * bit_len);
      memset(bitmap1_, 0, sizeof(BIT_TYPE) * bit_len);
      // Recursively build the node
      for (uint32_t i = 0, j = 0; i < ci->num_conflicts_; ++i) {
        uint32_t p = ci->positions_[i];
        uint32_t c = ci->conflicts_[i];
        bool branch = (c > 1) ? true : false;
        // find GRAIN, the range is [i, k)
        auto grain_size = c;
        auto base = PALIGN_DOWN(p, grain_kt);
        // auto base = (p / grain_kt) * grain_kt;
        auto k = i+1;
        while ((k < ci->num_conflicts_) && (ci->positions_[k] < (base + grain_kt))) {
          grain_size += ci->conflicts_[k];
          branch = branch | ((ci->conflicts_[k] > 1) ? true : false);
          k++;
        }
        // write keys
        if (grain_size == 0) {
          continue;
        } else if (!branch) {
          // for (int z = 0; z < grain_size; ++z) {
          //   printf("%d ", ks[j+z]);
          // }
          // printf("\n%d, %d, %d\n", i, k, grain_size);
          for (uint32_t ii = i; ii < k; ++ii) {
            uint32_t pp = ci->positions_[ii];
            set_entry_type(pp, Edge);
            keys_[pp] = ks[j++];
            if (ci->conflicts_[ii] == 0) {
              printf("error ci->conflicts_[ii] == 0 \n");
              exit(0);
            }
            // if (ks[j-1] == 0) {
            //   printf("error ks[j-1] == 0 \n");
            //   exit(0);
            // }
          }
          size_ += (k-i);
        } else if (grain_size < grain_kt) {
          set_entry_grain(base, Block, grain_kt);
          make_block(keys_ + base, ks+j, grain_size);
          j = j + grain_size;
        } else { // children
          // if next GRAIN's data size > threshold, then merge to a child
          uint32_t grain_num = 2;
          while (true) {
            uint32_t grain_size_next = 0;
            uint32_t k_in = k;
            while ((k_in < ci->num_conflicts_) && (ci->positions_[k_in] < (base + grain_kt * grain_num))) {
              grain_size_next += ci->conflicts_[k_in];
              k_in++;
            }
            // if (grain_size_next > hyper_para.max_bucket_size_ * (GRAIN / sizeof(Entry<KT, VT>))) {
            if (grain_size_next > grain_kt*8) {
              k = k_in;
              grain_size += grain_size_next;
              grain_num++;
            } else {
              grain_num--;
              break;
            }
          }

          entries_[base >> 1].child_ = new TNode<KT, VT>();
          for (auto b = 0; b < grain_num; ++b) {
            set_entry_grain(base + b * grain_kt, HTNode, grain_kt);
            entries_[(base + b * grain_kt) >> 1].child_ = entries_[base >> 1].child_;
          }
          // printf("%f %f\n", model_->slope_, model_->intercept_);
          // for (int z = 0; z < grain_size; ++z) {
          //   printf("%d ", ks[j+z]);
          // }
          // printf("\n");
          entries_[base >> 1].child_->build(ks + j, grain_size, depth + 1, hyper_para);
          j = j + grain_size;

          // origin
          // set_entry_grain(base, HTNode, grain_kt);
          // entries_[base >> 1].child_ = new TNode<KT, VT>();
          // // printf("%f %f\n", model_->slope_, model_->intercept_);
          // // for (int z = 0; z < grain_size; ++z) {
          // //   printf("%d ", ks[j+z]);
          // // }
          // // printf("\n");
          // entries_[base >> 1].child_->build(ks + j, grain_size, depth + 1, hyper_para);
          // j = j + grain_size;
        }
        i = k-1;
      }
      delete ci;
    }
  }

};

}
#endif
