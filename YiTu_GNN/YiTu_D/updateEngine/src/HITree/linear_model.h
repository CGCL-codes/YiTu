#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include "common.h"

namespace hitreespace {

template<class KT>
class LinearModel {
 public:
  double slope_;
  double intercept_;

  LinearModel() : slope_(0), intercept_(0) { }

  inline int64_t predict(KT key) const {
    return static_cast<int64_t>(std::floor(slope_ * key + intercept_));
  }

  inline double predict_double(KT key) const {
    return slope_ * static_cast<double>(key) + intercept_;
  }
};

template<class KT>
class LinearModelBuilder {
 public:
  int count_;
  double x_sum_;
  double y_sum_;
  double xx_sum_;
  double xy_sum_;
  KT x_min_;
  KT x_max_;
  double y_min_;
  double y_max_;

  LinearModelBuilder() : count_(0), x_sum_(0), y_sum_(0), xx_sum_(0), xy_sum_(0), 
                      x_min_(std::numeric_limits<KT>::max()), 
                      x_max_(std::numeric_limits<KT>::lowest()),
                      y_min_(std::numeric_limits<double>::max()), 
                      y_max_(std::numeric_limits<double>::lowest()) { }

  inline void add(KT x, double y) {
    count_++;
    x_sum_ += static_cast<double>(x);
    y_sum_ += static_cast<double>(y);
    xx_sum_ += static_cast<double>(x) * x;
    xy_sum_ += static_cast<double>(x) * y;
    x_min_ = std::min(x, x_min_);
    x_max_ = std::max(x, x_max_);
    y_min_ = std::min(y, y_min_);
    y_max_ = std::max(y, y_max_);
  }

  // TODO: the calculated slope or intercept is too small or too large, the 
  // precision is lost.
  void build(LinearModel<KT> *lrm) {
    if (count_ <= 1) {
      lrm->slope_ = 0;
      lrm->intercept_ = static_cast<double>(y_sum_);
      return;
    }

    if (static_cast<double>(count_) * xx_sum_ - x_sum_ * x_sum_ == 0) {
      // all values in a bucket have the same key.
      lrm->slope_ = 0;
      lrm->intercept_ = static_cast<double>(y_sum_) / count_;
      return;
    }

    auto slope = static_cast<double>(
        (static_cast<double>(count_) * xy_sum_ - x_sum_ * y_sum_) /
        (static_cast<double>(count_) * xx_sum_ - x_sum_ * x_sum_));
    auto intercept = static_cast<double>(
        (y_sum_ - static_cast<double>(slope) * x_sum_) / count_);
    lrm->slope_ = slope;
    lrm->intercept_ = intercept;

    // If floating point precision errors, fit spline
    if (lrm->slope_ <= 0) {
      lrm->slope_ = 1. * (y_max_ - y_min_) / (x_max_ - x_min_);
      lrm->intercept_ = -static_cast<double>(x_min_) * lrm->slope_;
    }
  }
};

}
#endif
