#pragma once

#include "tree.h"

template <typename SplitFunctionT, typename LeafFunctionT>
class Forest {    
public:
  Forest() {}
  virtual ~Forest() {}
  
  std::shared_ptr<LeafFunctionT> inferencest(const SamplePtr& sample) const {
    int n_trees = trees_.size();

    std::vector<std::shared_ptr<LeafFunctionT>> fcns;
    
    //inference of individual trees
    for(int tree_idx = 0; tree_idx < n_trees; ++tree_idx) {
      std::shared_ptr<LeafFunctionT> tree_fcn = trees_[tree_idx]->inference(sample);
      fcns.push_back(tree_fcn);
    }
    
    //combine tree fcns/results and collect all results
    return fcns[0]->Reduce(fcns);
  }  

  std::vector<std::shared_ptr<LeafFunctionT>> inferencemt(const std::vector<SamplePtr>& samples, int n_threads) const {
    std::vector<std::shared_ptr<LeafFunctionT>> targets(samples.size());

    omp_set_num_threads(n_threads);
    #pragma omp parallel for
    for(size_t sample_idx = 0; sample_idx < samples.size(); ++sample_idx) {
      targets[sample_idx] = inferencest(samples[sample_idx]);
    }

    return targets;
  }

  std::vector<std::shared_ptr<LeafFunctionT>> inferencemt(const std::vector<TrainDatum>& samples, int n_threads) const {
    std::vector<std::shared_ptr<LeafFunctionT>> targets(samples.size());

    omp_set_num_threads(n_threads);
    #pragma omp parallel for
    for(size_t sample_idx = 0; sample_idx < samples.size(); ++sample_idx) {
      targets[sample_idx] = inferencest(samples[sample_idx].sample);
    }

    return targets;
  }
  
  void AddTree(std::shared_ptr<Tree<SplitFunctionT, LeafFunctionT>> tree) { 
    trees_.push_back(tree); 
  }
  
  size_t trees_size() const { return trees_.size(); }
  // TreePtr trees(int idx) const { return trees_[idx]; }
  
  virtual void Save(SerializationOut& ar) const {
    size_t n_trees = trees_.size();
    std::cout << "[DEBUG] write " << n_trees << " trees" << std::endl;
    ar << n_trees;

    if(true) std::cout << "[Forest][write] write number of trees " << n_trees << std::endl;

    for(size_t tree_idx = 0; tree_idx < trees_.size(); ++tree_idx) {
      if(true) std::cout << "[Forest][write] write tree nb. " << tree_idx << std::endl;
      trees_[tree_idx]->Save(ar);
    }
  }

  virtual void Load(SerializationIn& ar) {
    size_t n_trees;
    ar >> n_trees;
    
    if(true) std::cout << "[Forest][read] nTrees: " << n_trees << std::endl;
    
    trees_.clear();
    for(size_t i = 0; i < n_trees; ++i) {
      if(true) std::cout << "[Forest][read] read tree " << (i+1) << " of " << n_trees << " - " << std::endl;
      
      auto tree = std::make_shared<Tree<SplitFunctionT, LeafFunctionT>>();
      tree->Load(ar);
      trees_.push_back(tree);
      
      if(true) std::cout << "[Forest][read] finished read tree " << (i+1) << " of " << n_trees << std::endl;
    }
  }

    
private:
  std::vector<std::shared_ptr<Tree<SplitFunctionT, LeafFunctionT>>> trees_;
};

