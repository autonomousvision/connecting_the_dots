#pragma once

#include <memory>

#include "serialization.h"
#include "leaffcn.h"
#include "splitfcn.h"

class Node {
public:
  Node() {}
  virtual ~Node() {}

  virtual std::shared_ptr<Node> Copy() const = 0;

  virtual int type() const = 0;

  virtual void Save(SerializationOut& ar) const = 0;
  virtual void Load(SerializationIn& ar) = 0;

};

typedef std::shared_ptr<Node> NodePtr;


template <typename LeafFunctionT>
class LeafNode : public Node {
public:
  static const int TYPE = 0;

  LeafNode() {}
  LeafNode(std::shared_ptr<LeafFunctionT> leaf_node_fcn) : leaf_node_fcn_(leaf_node_fcn) {}

  virtual ~LeafNode() {}

  virtual NodePtr Copy() const {
    auto node = std::make_shared<LeafNode>();
    node->leaf_node_fcn_ = leaf_node_fcn_->Copy();
    return node;
  }

  virtual void Save(SerializationOut& ar) const {
    leaf_node_fcn_->Save(ar);
  }

  virtual void Load(SerializationIn& ar) {
    leaf_node_fcn_ = std::make_shared<LeafFunctionT>();
    leaf_node_fcn_->Load(ar);
  }

  virtual int type() const { return TYPE; };
  std::shared_ptr<LeafFunctionT> leaf_node_fcn() const { return leaf_node_fcn_; }

private:
  std::shared_ptr<LeafFunctionT> leaf_node_fcn_;

DISABLE_COPY_AND_ASSIGN(LeafNode);
};


template <typename SplitFunctionT, typename LeafFunctionT>
class SplitNode : public Node {
public:
  static const int TYPE = 1;

  SplitNode() {}

  SplitNode(NodePtr left, NodePtr right, std::shared_ptr<SplitFunctionT> split_fcn) :
    left_(left), right_(right), split_fcn_(split_fcn)
  {}

  virtual ~SplitNode() {}

  virtual std::shared_ptr<Node> Copy() const {
    std::shared_ptr<SplitNode> node = std::make_shared<SplitNode>();
    node->left_ = left_->Copy();
    node->right_ = right_->Copy();
    node->split_fcn_ = split_fcn_->Copy();

    return node;
  }

  bool Split(SamplePtr sample) {
    return split_fcn_->Split(sample);
  }

  virtual void Save(SerializationOut& ar) const {
    split_fcn_->Save(ar);

    //left
    int type = left_->type();
    ar << type;
    left_->Save(ar);

    //right
    type = right_->type();
    ar << type;
    right_->Save(ar);
  }

  virtual void Load(SerializationIn& ar);


  virtual int type() const { return TYPE; }

  NodePtr left() const { return left_; }
  NodePtr right() const { return right_; }
  std::shared_ptr<SplitFunctionT> split_fcn() const { return split_fcn_; }

  void set_left(NodePtr left) { left_ = left; }
  void set_right(NodePtr right) { right_ = right; }
  void set_split_fcn(std::shared_ptr<SplitFunctionT> split_fcn) { split_fcn_ = split_fcn; }

public:
  NodePtr left_;
  NodePtr right_;
  std::shared_ptr<SplitFunctionT> split_fcn_;

DISABLE_COPY_AND_ASSIGN(SplitNode);
};


template <typename SplitFunctionT, typename LeafFunctionT>
NodePtr MakeNode(int type) {
  NodePtr node;
  if(type == LeafNode<LeafFunctionT>::TYPE) {
    node = std::make_shared<LeafNode<LeafFunctionT>>();
  }
  else if(type == SplitNode<SplitFunctionT, LeafFunctionT>::TYPE) {
    node = std::make_shared<SplitNode<SplitFunctionT, LeafFunctionT>>();
  }
  else {
    std::cout << "[ERROR] unknown node type" << std::endl;
    exit(-1);
  }

  return node;
}


template <typename SplitFunctionT, typename LeafFunctionT>
void SplitNode<SplitFunctionT, LeafFunctionT>::Load(SerializationIn& ar) {

  split_fcn_ = std::make_shared<SplitFunctionT>();
  split_fcn_->Load(ar);

  //left
  int left_type;
  ar >> left_type;
  left_ = MakeNode<SplitFunctionT, LeafFunctionT>(left_type);
  left_->Load(ar);

  //right
  int right_type;
  ar >> right_type;
  right_ = MakeNode<SplitFunctionT, LeafFunctionT>(right_type);
  right_->Load(ar);
}
