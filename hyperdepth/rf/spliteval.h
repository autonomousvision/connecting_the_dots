#pragma once

class SplitEvaluator {
public:
  SplitEvaluator(bool normalize)
    : normalize_(normalize) {}

  virtual ~SplitEvaluator() {}

  virtual float Eval(const std::vector<TrainDatum>& lefttargets, const std::vector<TrainDatum>& righttargets, int depth) const {
    float purity_left = Purity(lefttargets, depth);
    float purity_right = Purity(righttargets, depth);

    float normalize_left = 1.0;
    float normalize_right = 1.0;

    if(normalize_) {
      unsigned int n_left = lefttargets.size();
      unsigned int n_right = righttargets.size();
      unsigned int n_total = n_left + n_right;

      normalize_left = float(n_left) / float(n_total);
      normalize_right = float(n_right) / float(n_total);
    }

    float purity =  purity_left * normalize_left + purity_right * normalize_right;

    return purity;
  }

protected:
  virtual float Purity(const std::vector<TrainDatum>& targets, int depth) const = 0;

protected:
  bool normalize_;
};



class ClassificationIGSplitEvaluator : public SplitEvaluator {
public:
  ClassificationIGSplitEvaluator(bool normalize, int n_classes)
      : SplitEvaluator(normalize), n_classes_(n_classes) {}
  virtual ~ClassificationIGSplitEvaluator() {}

protected:
  virtual float Purity(const std::vector<TrainDatum>& targets, int depth) const {
    if(targets.size() == 0) return 0;

    std::vector<int> ps;
    ps.resize(n_classes_, 0);
    for(auto target : targets) {
      auto ctarget = std::static_pointer_cast<ClassificationTarget>(target.optimize_target);
      ps[ctarget->cl()] += 1;
    }

    float h = 0;
    for(int cl = 0; cl < n_classes_; ++cl) {
      float fi = float(ps[cl]) / float(targets.size());
      if(fi > 0) {
        h = h - fi * std::log(fi);
      }
    }

    return h;
  }

private:
  int n_classes_;
};

