#ifndef CONFLICTS_H
#define CONFLICTS_H

#include "common.h"
#include "linear_model.h"

namespace hitreespace {

struct ConflictsInfo {
  uint32_t* conflicts_;
  uint32_t* positions_;
  uint32_t num_conflicts_;
  uint32_t max_size_;
  uint32_t size_;

  ConflictsInfo(uint32_t size, uint32_t max_size) 
    : max_size_(max_size), num_conflicts_(0) {
    size_ = size;
    conflicts_ = new uint32_t[size];
    positions_ = new uint32_t[size];
  }

  ~ConflictsInfo() {
    delete[] conflicts_;
    delete[] positions_;
  }

  void add_conflict(uint32_t position, uint32_t conflict) {
    conflicts_[num_conflicts_] = conflict;
    positions_[num_conflicts_] = position;
    num_conflicts_ ++;
  }
};

template<typename KT, typename VT>
ConflictsInfo* build_linear_model(const KT* ks, uint32_t size,
                                  LinearModel<KT>*& model, 
                                  double size_amp) {
  if (model != nullptr) {
    model->slope_ = model->intercept_ = 0;
  } else {
    model = new LinearModel<KT>();
  }
  // OPT: Find a linear regression model that has the minimum conflict degree
  // The heuristics below is a simple method that scales the positions
  KT min_key = ks[0];
  KT max_key = ks[size - 1];
  KT key_space = max_key - min_key;
  if (compare(min_key, max_key)) {
    delete model;
    model = nullptr;
    return nullptr;
  }
  uint32_t max_size = static_cast<uint32_t>(size * size_amp);
  max_size = PALIGN_UP(max_size, GRAIN / sizeof(KT));
  LinearModelBuilder<KT> builder;
  uint32_t j = 0;
  for (uint32_t i = 0; i < size; ++ i) {
    KT key = ks[i];
    // double y = max_size * (key - min_key) / key_space;
    double y = i;
    builder.add(key, y);
  }
  builder.build(model);
  if (compare(model->slope_, 0.)) {
    // Fail to build a linear model
    delete model;
    model = nullptr;
    return nullptr;
  } else {
    // model->intercept_ = -model->slope_ * (min_key) + 0.5;
    // int64_t predicted_size = model->predict(max_key) + 1;
    // printf("%d, %d\n", predicted_size, max_size);
    // if (predicted_size > 1) {
    //   max_size = std::min(predicted_size, static_cast<int64_t>(max_size));
    // }
    // model->intercept_ = -model->slope_ * (min_key) + 0.5;
    int64_t predicted_size = model->predict(max_key) + 1;
    // printf("%lf, %lf, %d, %d\n", model->slope_, model->intercept_, predicted_size, max_size);
    if (predicted_size > 1) {
      model->slope_ *= (1.0 * max_size / predicted_size);
      model->intercept_ = -model->slope_ * (min_key) + 0.5;
      // max_size = std::min(predicted_size, static_cast<int64_t>(max_size));
    } else {
      // printf("predicted_size = %lld\n", predicted_size);
    }
    // if (predicted_size < size * 1.2) {
    //   model->slope_ *= (1.0 * max_size / predicted_size);
    //   model->intercept_ = -model->slope_ * (min_key) + 0.5;
    // } else {
    //   // model->intercept_ = -model->slope_ * (min_key) + 0.5;
    //   max_size = std::min(predicted_size, static_cast<int64_t>(max_size));
    // }
    uint32_t first_pos = std::min(std::max(model->predict(min_key), 0L), 
                                  static_cast<int64_t>(max_size - 1));
    uint32_t last_pos = std::min(std::max(model->predict(max_key), 0L), 
                                  static_cast<int64_t>(max_size - 1));
    if (last_pos == first_pos) {
      // Model fails to predict since all predicted positions are rounded to the 
      // same one
      model->slope_ = size / (max_key - min_key);
      model->intercept_ = -model->slope_ * (min_key) + 0.5;
    }
    ConflictsInfo* ci = new ConflictsInfo(size, max_size);
    uint32_t p_last = first_pos;
    uint32_t conflict = 1;
    for (uint32_t i = 1; i < size; ++ i) {
      uint32_t p = std::min(std::max(model->predict(ks[i]), 0L), 
                                    static_cast<int64_t>(max_size - 1));
      if (p == p_last) {
        conflict ++;
      } else {
        ci->add_conflict(p_last, conflict);
        p_last = p;
        conflict = 1;
      }
    }
    if (conflict > 0) {
      ci->add_conflict(p_last, conflict);
    }
    return ci;
  }
}

template<typename KT, typename VT>
uint32_t compute_tail_conflicts(const KT* ks, uint32_t size, 
                                double size_amp, float kTailPercent=0.99) {
  // The input keys should be ordered
  LinearModel<KT>* model = new LinearModel<KT>();
  ConflictsInfo* ci = build_linear_model<KT, VT>(ks, size, model, size_amp);
  delete model;

  if (ci == nullptr) {
    return 0;
  } else if (ci->num_conflicts_ == 0) {
    // printf("size = %d %f\n", size, size_amp);
    delete ci;
    return 0;
  } else {
    std::sort(ci->conflicts_, ci->conflicts_ + ci->num_conflicts_);
    uint32_t tail_conflicts = ci->conflicts_[std::max(0, int(ci->num_conflicts_ * kTailPercent) - 1)];
    delete ci;
    return tail_conflicts - 1;
  }
}

}
#endif
