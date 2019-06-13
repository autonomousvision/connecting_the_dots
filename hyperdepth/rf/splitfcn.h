#pragma once

#include <random>
    
class SplitFunction {
public:
  SplitFunction() {}
  virtual ~SplitFunction() {}

  virtual float Compute(SamplePtr sample) const = 0;
  
  virtual bool Split(SamplePtr sample) const {
    return Compute(sample) < threshold_;
  }
  
  virtual void Save(SerializationOut& ar) const {
    ar << threshold_;
  }

  virtual void Load(SerializationIn& ar) {
    ar >> threshold_;
  }
  
  virtual float threshold() const { return threshold_; }
  virtual void set_threshold(float threshold) { threshold_ = threshold; }
    
protected:
  float threshold_;
};



class SplitFunctionPixelDifference : public SplitFunction {
public:

  SplitFunctionPixelDifference() {}
  virtual ~SplitFunctionPixelDifference() {}
  
  virtual std::shared_ptr<SplitFunctionPixelDifference> Copy() const {
    std::shared_ptr<SplitFunctionPixelDifference> split_fcn = std::make_shared<SplitFunctionPixelDifference>();
    split_fcn->threshold_ = threshold_;
    split_fcn->c0_ = c0_;
    split_fcn->c1_ = c1_;
    split_fcn->h0_ = h0_;
    split_fcn->h1_ = h1_;
    split_fcn->w0_ = w0_;
    split_fcn->w1_ = w1_;

    return split_fcn;
  }

  virtual std::shared_ptr<SplitFunctionPixelDifference> Generate(std::mt19937& rng, const SamplePtr sample) const {
    std::shared_ptr<SplitFunctionPixelDifference> split_fcn = std::make_shared<SplitFunctionPixelDifference>();

    std::uniform_int_distribution<int> cdist(0, sample->channels()-1);
    split_fcn->c0_ = cdist(rng);
    split_fcn->c1_ = cdist(rng);
    
    std::uniform_int_distribution<int> hdist(0, sample->height()-1);
    split_fcn->h0_ = hdist(rng);
    split_fcn->h1_ = hdist(rng);
    
    std::uniform_int_distribution<int> wdist(0, sample->width()-1);
    split_fcn->w0_ = wdist(rng);
    split_fcn->w1_ = wdist(rng);
    
    return split_fcn;
  }
  
  virtual float Compute(SamplePtr sample) const {
    return (*sample)(c0_, h0_, w0_) - (*sample)(c1_, h1_, w1_);
  }
  
  virtual void Save(SerializationOut& ar) const {
    SplitFunction::Save(ar);
    ar << c0_;
    ar << c1_;
    ar << h0_;
    ar << h1_;
    ar << w0_;
    ar << w1_;
  }
  
  virtual void Load(SerializationIn& ar) {
    SplitFunction::Load(ar);

    ar >> c0_;
    ar >> c1_;
    ar >> h0_;
    ar >> h1_;
    ar >> w0_;
    ar >> w1_;
  }
  
private:
  int c0_;
  int c1_;
  int h0_;
  int h1_;
  int w0_;
  int w1_;

DISABLE_COPY_AND_ASSIGN(SplitFunctionPixelDifference);
};


