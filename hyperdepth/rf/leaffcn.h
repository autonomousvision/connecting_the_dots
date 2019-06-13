#pragma once

#include <iostream>

#include "common.h"
#include "data.h"


class ClassProbabilitiesLeafFunction {
public:
  ClassProbabilitiesLeafFunction() : n_classes_(-1) {}
  ClassProbabilitiesLeafFunction(int n_classes) : n_classes_(n_classes) {}
  virtual ~ClassProbabilitiesLeafFunction() {}

  virtual std::shared_ptr<ClassProbabilitiesLeafFunction> Copy() const {
    auto fcn = std::make_shared<ClassProbabilitiesLeafFunction>();
    fcn->n_classes_ = n_classes_;
    fcn->counts_.resize(counts_.size());
    for(size_t idx = 0; idx < counts_.size(); ++idx) {
      fcn->counts_[idx] = counts_[idx];
    }
    fcn->sum_counts_ = sum_counts_;

    return fcn;
  }

  virtual std::shared_ptr<ClassProbabilitiesLeafFunction> Create(const std::vector<TrainDatum>& samples) {
    auto stat = std::make_shared<ClassProbabilitiesLeafFunction>();

    stat->counts_.resize(n_classes_, 0);
    for(auto sample : samples) {
      auto ctarget = std::static_pointer_cast<ClassificationTarget>(sample.target);
      stat->counts_[ctarget->cl()] += 1;
    }
    stat->sum_counts_ = samples.size();

    return stat;
  }

  virtual std::shared_ptr<ClassProbabilitiesLeafFunction> Reduce(const std::vector<std::shared_ptr<ClassProbabilitiesLeafFunction>>& fcns) const {
    auto stat = std::make_shared<ClassProbabilitiesLeafFunction>();
    auto cfcn0 = std::static_pointer_cast<ClassProbabilitiesLeafFunction>(fcns[0]);
    stat->counts_.resize(cfcn0->counts_.size(), 0);
    stat->sum_counts_ = 0;

    for(auto fcn : fcns) {
      auto cfcn = std::static_pointer_cast<ClassProbabilitiesLeafFunction>(fcn);
      for(size_t cl = 0; cl < stat->counts_.size(); ++cl) {
        stat->counts_[cl] += cfcn->counts_[cl];
      }
      stat->sum_counts_ += cfcn->sum_counts_;
    }

    return stat;
  }

  virtual int argmax() const {
    int max_idx = 0;
    int max_count = counts_[0];
    for(size_t idx = 1; idx < counts_.size(); ++idx) {
      if(counts_[idx] > max_count) {
        max_count = counts_[idx];
        max_idx = idx;
      }
    }
    return max_idx;
  }

  virtual void Save(SerializationOut& ar) const {
    ar << n_classes_;
    int n_counts = counts_.size();
    ar << n_counts;
    for(int idx = 0; idx < n_counts; ++idx) {
      ar << counts_[idx];
    }
    ar << sum_counts_;
  }

  virtual void Load(SerializationIn& ar) {
    ar >> n_classes_;
    int n_counts;
    ar >> n_counts;
    counts_.resize(n_counts);
    for(int idx = 0; idx < n_counts; ++idx) {
      ar >> counts_[idx];
    }
    ar >> sum_counts_;
  }

public:
  int n_classes_;

  std::vector<int> counts_;
  int sum_counts_;

DISABLE_COPY_AND_ASSIGN(ClassProbabilitiesLeafFunction);
};


