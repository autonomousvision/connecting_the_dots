#pragma once

#include <chrono>
#include <set>
#include <queue>

#include "threadpool.h"

#include "forest.h"
#include "spliteval.h"


enum class TrainType : int {
  TRAIN = 0,
  RETRAIN = 1,
  RETRAIN_WITH_REPLACEMENT = 2
};

struct TrainParameters {
  TrainType train_type;
  int n_trees;
  int max_tree_depth;
  int n_test_split_functions;
  int n_test_thresholds;
  int n_test_samples;
  int min_samples_to_split;
  int min_samples_for_leaf;
  int print_node_info;

  TrainParameters() :
    train_type(TrainType::TRAIN),
    n_trees(5),
    max_tree_depth(7),
    n_test_split_functions(50),
    n_test_thresholds(10),
    n_test_samples(100),
    min_samples_to_split(14),
    min_samples_for_leaf(7),
    print_node_info(100)
  {}
};


template <typename SplitFunctionT, typename LeafFunctionT, typename SplitEvaluatorT>
class TrainForest {
public:
   TrainForest(const TrainParameters& params, const std::shared_ptr<SplitFunctionT> gen_split_fcn, const std::shared_ptr<LeafFunctionT> gen_leaf_fcn, const std::shared_ptr<SplitEvaluatorT> split_eval, int n_threads, bool verbose)
     : params_(params), gen_split_fcn_(gen_split_fcn), gen_leaf_fcn_(gen_leaf_fcn), split_eval_(split_eval), n_threads(n_threads), verbose_(verbose) {

    n_created_nodes_ = 0;
    n_max_nodes_ = 1;
    unsigned long n_nodes_d = 1;
    for(int depth = 0; depth < params.max_tree_depth; ++depth) {
      n_nodes_d *= 2;
      n_max_nodes_ += n_nodes_d;
    }
    n_max_nodes_ *= params.n_trees;
  }

  virtual ~TrainForest() {}

  virtual std::shared_ptr<Forest<SplitFunctionT, LeafFunctionT>> Train(const std::vector<TrainDatum>& samples, TrainType train_type, const std::shared_ptr<Forest<SplitFunctionT, LeafFunctionT>>& old_forest) = 0;

protected:
  virtual void PrintParams() {
    if(verbose_){
      #pragma omp critical (TrainForest_train)
      {
        std::cout << "[TRAIN] training forest " << std::endl;
        std::cout << "[TRAIN] n_trees               : " << params_.n_trees << std::endl;
        std::cout << "[TRAIN] max_tree_depth        : " << params_.max_tree_depth << std::endl;
        std::cout << "[TRAIN] n_test_split_functions: " << params_.n_test_split_functions << std::endl;
        std::cout << "[TRAIN] n_test_thresholds     : " << params_.n_test_thresholds << std::endl;
        std::cout << "[TRAIN] n_test_samples        : " << params_.n_test_samples << std::endl;
        std::cout << "[TRAIN] min_samples_to_split  : " << params_.min_samples_to_split << std::endl;
      }
    }
  }

  virtual void UpdateNodeInfo(unsigned int depth, bool leaf) {
    if(verbose_) {
      n_created_nodes_ += 1;

      if(leaf) {
        unsigned long n_nodes_d = 1;
        unsigned int n_remove_max_nodes = 0;
        for(int d = depth; d < params_.max_tree_depth; ++d) {
          n_nodes_d *= 2;
          n_remove_max_nodes += n_nodes_d;
        }
        n_max_nodes_ -= n_remove_max_nodes;
      }

      if(n_created_nodes_ % params_.print_node_info == 0 || n_created_nodes_ == n_max_nodes_) {
        std::cout << "[Forest]"
          << " created node number " << n_created_nodes_
          << " @ depth " << depth
          << ", max. " << n_max_nodes_ << " left"
          << " => " << (double(n_created_nodes_) / double(n_max_nodes_))
          << " done" << std::endl;
      }
    }
  }

  virtual void SampleData(const std::vector<TrainDatum>& all, std::vector<TrainDatum>& sampled, std::mt19937& rng) {
    unsigned int n = all.size();
    unsigned int k = params_.n_test_samples;
    k = n < k ? n : k;

    std::set<int> indices;
    std::uniform_int_distribution<int> udist(0, all.size()-1);
    while(indices.size() < k) {
      int idx = udist(rng);
      indices.insert(idx);
    }

    sampled.resize(k);
    int sidx = 0;
    for(int idx : indices) {
      sampled[sidx] = all[idx];
      sidx += 1;
    }
  }

  virtual void Split(const std::shared_ptr<SplitFunctionT>& split_function, const std::vector<TrainDatum>& samples, std::vector<TrainDatum>& left, std::vector<TrainDatum>& right) {
    for(auto sample : samples) {
      if(split_function->Split(sample.sample)) {
        left.push_back(sample);
      }
      else {
        right.push_back(sample);
      }
    }
  }


  virtual std::shared_ptr<SplitFunctionT> OptimizeSplitFunction(const std::vector<TrainDatum>& samples, int depth, std::mt19937& rng) {
    std::vector<TrainDatum> split_samples;
    SampleData(samples, split_samples, rng);

    unsigned int min_samples_for_leaf = params_.min_samples_for_leaf;

    float min_cost = std::numeric_limits<float>::max();
    std::shared_ptr<SplitFunctionT> best_split_fcn;
    float best_threshold = 0;

    for(int split_fcn_idx = 0; split_fcn_idx < params_.n_test_split_functions; ++split_fcn_idx) {
      auto split_fcn = gen_split_fcn_->Generate(rng, samples[0].sample);

      for(int threshold_idx = 0; threshold_idx < params_.n_test_thresholds; ++threshold_idx) {
        std::uniform_int_distribution<int> udist(0, split_samples.size()-1);
        int rand_split_sample_idx = udist(rng);
        float threshold = split_fcn->Compute(split_samples[rand_split_sample_idx].sample);
        split_fcn->set_threshold(threshold);

        std::vector<TrainDatum> left;
        std::vector<TrainDatum> right;
        Split(split_fcn, split_samples, left, right);
        if(left.size() < min_samples_for_leaf || right.size() < min_samples_for_leaf) {
          continue;
        }

        // std::cout << "split done " << left.size() << "," << right.size() << std::endl;
        float split_cost = split_eval_->Eval(left, right, depth);
        // std::cout << ", " << split_cost << ", " << threshold << "; " << std::endl;

        if(split_cost < min_cost) {
          min_cost = split_cost;
          best_split_fcn = split_fcn;
          best_threshold = threshold; //need theshold extra because of pointer
        }
      }
    }

    if(best_split_fcn != nullptr) {
      best_split_fcn->set_threshold(best_threshold);
    }

    return best_split_fcn;
  }


  virtual NodePtr CreateLeafNode(const std::vector<TrainDatum>& samples, unsigned int depth) {
    auto leaf_fct = gen_leaf_fcn_->Create(samples);
    auto node = std::make_shared<LeafNode<LeafFunctionT>>(leaf_fct);

    UpdateNodeInfo(depth, true);

    return node;
  }

protected:
  const TrainParameters& params_;
  const std::shared_ptr<SplitFunctionT> gen_split_fcn_;
  const std::shared_ptr<LeafFunctionT> gen_leaf_fcn_;
  const std::shared_ptr<SplitEvaluatorT> split_eval_;
  int n_threads;
  bool verbose_;

  unsigned long n_created_nodes_;
  unsigned long n_max_nodes_;
};


template <typename SplitFunctionT, typename LeafFunctionT, typename SplitEvaluatorT>
class TrainForestRecursive : public TrainForest<SplitFunctionT, LeafFunctionT, SplitEvaluatorT> {
public:
  TrainForestRecursive(const TrainParameters& params, const std::shared_ptr<SplitFunctionT> gen_split_fcn, const std::shared_ptr<LeafFunctionT> gen_leaf_fcn, const std::shared_ptr<SplitEvaluatorT> split_eval, int n_threads, bool verbose)
    : TrainForest<SplitFunctionT, LeafFunctionT, SplitEvaluatorT>(params, gen_split_fcn, gen_leaf_fcn, split_eval, n_threads, verbose) {}

  virtual ~TrainForestRecursive() {}

  virtual std::shared_ptr<Forest<SplitFunctionT, LeafFunctionT>> Train(const std::vector<TrainDatum>& samples, TrainType train_type, const std::shared_ptr<Forest<SplitFunctionT, LeafFunctionT>>& old_forest) {

    this->PrintParams();

    auto tim = std::chrono::system_clock::now();
    auto forest = std::make_shared<Forest<SplitFunctionT, LeafFunctionT>>();

    omp_set_num_threads(this->n_threads);
    #pragma omp parallel for ordered
    for(size_t treeIdx = 0; treeIdx < this->params_.n_trees; ++treeIdx) {
      auto treetim = std::chrono::system_clock::now();

      #pragma omp critical (TrainForest_train)
      {
        if(this->verbose_){
          std::cout << "[TRAIN][START] training tree " << treeIdx << " of " << this->params_.n_trees << std::endl;
        }
      }

      std::shared_ptr<Tree<SplitFunctionT, LeafFunctionT>> old_tree;
      if(old_forest != 0 && treeIdx < old_forest->trees_size()) {
        old_tree = old_forest->trees(treeIdx);
      }

      std::random_device rd;
      std::mt19937 rng(rd());

      auto tree = Train(samples, train_type, old_tree,rng);

      #pragma omp critical (TrainForest_train)
      {
        forest->AddTree(tree);
        if(this->verbose_){
          auto now = std::chrono::system_clock::now();
          auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - treetim);
          std::cout << "[TRAIN][FINISHED] training tree " << treeIdx << " of " << this->params_.n_trees << " - took " << (ms.count() * 1e-3) << "[s]" << std::endl;
          std::cout << "[TRAIN][FINISHED] " << (this->params_.n_trees - forest->trees_size()) << " left for training" << std::endl;
        }
      }
    }

    if(this->verbose_){
      auto now = std::chrono::system_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - tim);
      std::cout << "[TRAIN][FINISHED] training forest - took " << (ms.count() * 1e-3) << "[s]" << std::endl;
    }

    return forest;
  }

private:
  virtual std::shared_ptr<Tree<SplitFunctionT, LeafFunctionT>> Train(const std::vector<TrainDatum>& samples, TrainType train_type, const std::shared_ptr<Tree<SplitFunctionT, LeafFunctionT>>& old_tree, std::mt19937& rng) {
    NodePtr old_root;
    if(old_tree != nullptr) {
      old_root = old_tree->root();
    }

    NodePtr root = Train(samples, train_type, old_root, 0, rng);
    return std::make_shared<Tree<SplitFunctionT, LeafFunctionT>>(root);
  }

  virtual NodePtr Train(const std::vector<TrainDatum>& samples, TrainType train_type, const NodePtr& old_node, unsigned int depth, std::mt19937& rng) {

    if(depth < this->params_.max_tree_depth && samples.size() > this->params_.min_samples_to_split) {
      std::shared_ptr<SplitFunctionT> best_split_fcn;
      bool was_split_node = false;
      if(old_node == nullptr || old_node->type() == LeafNode<LeafFunctionT>::TYPE) {
        best_split_fcn = this->OptimizeSplitFunction(samples, depth, rng);
        was_split_node = false;
      }
      else if(old_node->type() == SplitNode<SplitFunctionT, LeafFunctionT>::TYPE) {
        auto split_node = std::static_pointer_cast<SplitNode<SplitFunctionT, LeafFunctionT>>(old_node);
        best_split_fcn = split_node->split_fcn()->Copy();
        was_split_node = true;
      }

      if(best_split_fcn == nullptr) {
        if(old_node == nullptr || train_type == TrainType::TRAIN || train_type == TrainType::RETRAIN_WITH_REPLACEMENT) {
          return this->CreateLeafNode(samples, depth);
        }
        else if(train_type == TrainType::RETRAIN) {
          return old_node->Copy();
        }
        else {
          std::cout << "[ERROR] unknown train type" << std::endl;
          exit(-1);
        }
      }

      // (1) split samples
      std::vector<TrainDatum> leftsamples, rightsamples;
      this->Split(best_split_fcn, samples, leftsamples, rightsamples);

      //output node information
      this->UpdateNodeInfo(depth, false);

      //create split node - recursively train the siblings
      if(was_split_node) {
        auto split_node = std::static_pointer_cast<SplitNode<SplitFunctionT, LeafFunctionT>>(old_node);
        NodePtr left = this->Train(leftsamples, train_type, split_node->left(), depth + 1, rng);
        NodePtr right = this->Train(rightsamples, train_type, split_node->right(), depth + 1, rng);
        auto new_node = std::make_shared<SplitNode<SplitFunctionT, LeafFunctionT>>(left, right, best_split_fcn);
        return new_node;
      }
      else {
        NodePtr left = this->Train(leftsamples, train_type, nullptr, depth + 1, rng);
        NodePtr right = this->Train(rightsamples, train_type, nullptr, depth + 1, rng);
        auto new_node = std::make_shared<SplitNode<SplitFunctionT, LeafFunctionT>>(left, right, best_split_fcn);
        return new_node;
      }
    } // if samples < min_samples || depth >= max_depth then make leaf node
    else {
      if(old_node == 0 || train_type == TrainType::TRAIN || train_type == TrainType::RETRAIN_WITH_REPLACEMENT) {
        return this->CreateLeafNode(samples, depth);
      }
      else if(train_type == TrainType::RETRAIN) {
        return old_node->Copy();
      }
      else {
        std::cout << "[ERROR] unknown train type" << std::endl;
        exit(-1);
      }
    }
  }

};


struct QueueTuple {
  int depth;
  std::vector<TrainDatum> train_data;
  NodePtr* parent;

  QueueTuple() : depth(-1), train_data(), parent(nullptr) {}
  QueueTuple(int depth, std::vector<TrainDatum> train_data, NodePtr* parent) :
    depth(depth), train_data(train_data), parent(parent) {}
};

template <typename SplitFunctionT, typename LeafFunctionT, typename SplitEvaluatorT>
class TrainForestQueued : public TrainForest<SplitFunctionT, LeafFunctionT, SplitEvaluatorT> {
public:
  TrainForestQueued(const TrainParameters& params, const std::shared_ptr<SplitFunctionT> gen_split_fcn, const std::shared_ptr<LeafFunctionT> gen_leaf_fcn, const std::shared_ptr<SplitEvaluatorT> split_eval, int n_threads, bool verbose)
    : TrainForest<SplitFunctionT, LeafFunctionT, SplitEvaluatorT>(params, gen_split_fcn, gen_leaf_fcn, split_eval, n_threads, verbose) {}

  virtual ~TrainForestQueued() {}

  virtual std::shared_ptr<Forest<SplitFunctionT, LeafFunctionT>> Train(const std::vector<TrainDatum>& samples, TrainType train_type, const std::shared_ptr<Forest<SplitFunctionT, LeafFunctionT>>& old_forest) {
    this->PrintParams();

    auto tim = std::chrono::system_clock::now();
    auto forest = std::make_shared<Forest<SplitFunctionT, LeafFunctionT>>();

    std::cout << "[TRAIN] create pool with " << this->n_threads << " threads" << std::endl;
    auto pool = std::make_shared<ThreadPool>(this->n_threads);
    for(int treeidx = 0; treeidx < this->params_.n_trees; ++treeidx) {
      auto tree = std::make_shared<Tree<SplitFunctionT, LeafFunctionT>>();
      forest->AddTree(tree);
      AddJob(pool, QueueTuple(0, samples, &(tree->root_)));
    }

    while(pool->has_running_tasks()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if(this->verbose_){
      auto now = std::chrono::system_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - tim);
      std::cout << "[TRAIN][FINISHED] training forest - took " << (ms.count() * 1e-3) << "[s]" << std::endl;
    }

    return forest;
  }

private:
  virtual void AddJob(std::shared_ptr<ThreadPool> pool, QueueTuple data) {
    pool->enqueue([this](std::shared_ptr<ThreadPool> pool, QueueTuple data) {
      std::random_device rd;
      std::mt19937 rng(rd());

      std::shared_ptr<SplitFunctionT> best_split_fcn = nullptr;

      if(data.depth < this->params_.max_tree_depth && int(data.train_data.size()) > this->params_.min_samples_to_split) {
        best_split_fcn = this->OptimizeSplitFunction(data.train_data, data.depth, rng);
      }

      if(best_split_fcn == nullptr) {
        auto node = this->CreateLeafNode(data.train_data, data.depth);
        *(data.parent) = node;
      }
      else {
        this->UpdateNodeInfo(data.depth, false);
        auto node = std::make_shared<SplitNode<SplitFunctionT, LeafFunctionT>>();
        node->split_fcn_ = best_split_fcn;
        *(data.parent) = node;

        QueueTuple left;
        QueueTuple right;
        this->Split(best_split_fcn, data.train_data, left.train_data, right.train_data);

        left.depth = data.depth + 1;
        right.depth = data.depth + 1;

        left.parent = &(node->left_);
        right.parent = &(node->right_);

        this->AddJob(pool, left);
        this->AddJob(pool, right);
      }
    }, pool, data);
  }
};
