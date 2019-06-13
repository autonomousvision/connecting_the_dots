#pragma once

#include "node.h"

template <typename SplitFunctionT, typename LeafFunctionT>
class Tree {    
public:
  Tree() : root_(nullptr) {}
  Tree(NodePtr root) : root_(root) {}
  
  virtual ~Tree() {}
  
  std::shared_ptr<LeafFunctionT> inference(const SamplePtr sample) const {
    if(root_ == nullptr) {
      std::cout << "[ERROR] tree inference root node is NULL";
      exit(-1);
    }
    
    NodePtr node = root_;
    while(node->type() == SplitNode<SplitFunctionT, LeafFunctionT>::TYPE) {
      auto splitNode = std::static_pointer_cast<SplitNode<SplitFunctionT, LeafFunctionT>>(node);
      bool left = splitNode->Split(sample);
      if(left) {
        node = splitNode->left();
      }
      else {
        node = splitNode->right();
      }
    }
    
    auto leaf_node = std::static_pointer_cast<LeafNode<LeafFunctionT>>(node);
    return leaf_node->leaf_node_fcn();
  }
  
  NodePtr root() const { return root_; }
  void set_root(NodePtr root) { root_ = root; }
  
  virtual void Save(SerializationOut& ar) const {
    int type = root_->type();
    ar << type;
    root_->Save(ar);
  }
  
  virtual void Load(SerializationIn& ar) {
    int type;
    ar >> type;
    root_ = MakeNode<SplitFunctionT, LeafFunctionT>(type);
    root_->Load(ar);
  }
    
    
public:
  NodePtr root_;
};

