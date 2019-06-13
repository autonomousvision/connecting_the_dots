#pragma once

#include <vector>


class Sample {        
public:
  Sample(int channels, int height, int width) 
    : channels_(channels), height_(height), width_(width) {}
  
  virtual ~Sample() {}
  
  virtual float at(int c, int h, int w) const = 0;

  virtual float operator()(int c, int h, int w) const {
    return at(c,h,w);
  }

  virtual int channels() const { return channels_; }
  virtual int height() const { return height_; }
  virtual int width() const { return width_; }

protected:
  int channels_;
  int height_;
  int width_;
};

typedef std::shared_ptr<Sample> SamplePtr;




class Target {
public:
  Target() {}
  virtual ~Target() {}
};

typedef std::shared_ptr<Target> TargetPtr;
typedef std::vector<TargetPtr> VecTargetPtr;
typedef std::shared_ptr<VecTargetPtr> VecPtrTargetPtr;


class ClassificationTarget : public Target {
public:
  ClassificationTarget(int cl) : cl_(cl) {}
  virtual ~ClassificationTarget() {}
  int cl() const { return cl_; }

private:
  int cl_;
};

typedef std::shared_ptr<ClassificationTarget> ClassificationTargetPtr;




struct TrainDatum {
  SamplePtr sample;
  TargetPtr target;
  TargetPtr optimize_target;
  
  TrainDatum() : sample(nullptr), target(nullptr), optimize_target(nullptr) {}

  TrainDatum(SamplePtr sample, TargetPtr target) 
    : sample(sample), target(target), optimize_target(target) {}

  TrainDatum(SamplePtr sample, TargetPtr target, TargetPtr optimize_target) 
    : sample(sample), target(target), optimize_target(optimize_target) {}
};
