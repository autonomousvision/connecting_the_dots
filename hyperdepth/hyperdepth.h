#include <sstream>
#include <iomanip>

#include "rf/forest.h"
#include "rf/spliteval.h"

class HyperdepthSplitEvaluator : public SplitEvaluator {
public:
  HyperdepthSplitEvaluator(bool normalize, int n_classes, int n_disp_bins, int depth_switch)
      : SplitEvaluator(normalize), n_classes_(n_classes), n_disp_bins_(n_disp_bins), depth_switch_(depth_switch) {}
  virtual ~HyperdepthSplitEvaluator() {}

protected:
  virtual float Purity(const std::vector<TrainDatum>& targets, int depth) const {
    if(targets.size() == 0) return 0;

    int n_classes = n_classes_;
    if(depth >= depth_switch_) {
      n_classes *= n_disp_bins_;
    }

    std::vector<int> ps;
    ps.resize(n_classes, 0);
    for(auto target : targets) {
      auto ctarget = std::static_pointer_cast<ClassificationTarget>(target.optimize_target);
      int cl = ctarget->cl();
      if(depth < depth_switch_) {
        cl /= n_disp_bins_;
      }
      ps[cl] += 1;
    }

    float h = 0;
    for(int cl = 0; cl < n_classes; ++cl) {
      float fi = float(ps[cl]) / float(targets.size());
      if(fi > 0) {
        h = h - fi * std::log(fi);
      }
    }

    return h;
  }

private:
  int n_classes_;
  int n_disp_bins_;
  int depth_switch_;
};


class HyperdepthLeafFunction {
public:
  HyperdepthLeafFunction() : n_classes_(-1) {}
  HyperdepthLeafFunction(int n_classes) : n_classes_(n_classes) {}
  virtual ~HyperdepthLeafFunction() {}

  virtual std::shared_ptr<HyperdepthLeafFunction> Copy() const {
    auto fcn = std::make_shared<HyperdepthLeafFunction>();
    fcn->n_classes_ = n_classes_;
    fcn->counts_.resize(counts_.size());
    for(size_t idx = 0; idx < counts_.size(); ++idx) {
      fcn->counts_[idx] = counts_[idx];
    }
    fcn->sum_counts_ = sum_counts_;

    return fcn;
  }

  virtual std::shared_ptr<HyperdepthLeafFunction> Create(const std::vector<TrainDatum>& samples) {
    auto stat = std::make_shared<HyperdepthLeafFunction>();

    stat->counts_.resize(n_classes_, 0);
    for(auto sample : samples) {
      auto ctarget = std::static_pointer_cast<ClassificationTarget>(sample.target);
      stat->counts_[ctarget->cl()] += 1;
    }
    stat->sum_counts_ = samples.size();

    return stat;
  }

  virtual std::shared_ptr<HyperdepthLeafFunction> Reduce(const std::vector<std::shared_ptr<HyperdepthLeafFunction>>& fcns) const {
    auto stat = std::make_shared<HyperdepthLeafFunction>();
    auto cfcn0 = std::static_pointer_cast<HyperdepthLeafFunction>(fcns[0]);
    stat->counts_.resize(cfcn0->counts_.size(), 0);
    stat->sum_counts_ = 0;

    for(auto fcn : fcns) {
      auto cfcn = std::static_pointer_cast<HyperdepthLeafFunction>(fcn);
      for(size_t cl = 0; cl < stat->counts_.size(); ++cl) {
        stat->counts_[cl] += cfcn->counts_[cl];
      }
      stat->sum_counts_ += cfcn->sum_counts_;
    }

    return stat;
  }

  virtual std::tuple<int,int> argmax() const {
    int max_idx = 0;
    int max_count = counts_[0];
    int max2_idx = -1;
    int max2_count = -1;
    for(size_t idx = 1; idx < counts_.size(); ++idx) {
      if(counts_[idx] > max_count) {
        max2_count = max_count;
        max2_idx = max_idx;
        max_count = counts_[idx];
        max_idx = idx;
      }
      else if(counts_[idx] > max2_count) {
        max2_count = counts_[idx];
        max2_idx = idx;
      }
    }
    return std::make_tuple(max_idx, max2_idx);
  }

  virtual std::vector<float> prob_vec() const {
    std::vector<float> probs(counts_.size(), 0.f);
    int sum = 0;
    for(int cnt : counts_) {
      sum += cnt;
    }
    for(size_t idx = 0; idx < counts_.size(); ++idx) {
      probs[idx] = float(counts_[idx]) / sum;
    }
    return probs;
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

DISABLE_COPY_AND_ASSIGN(HyperdepthLeafFunction);
};


typedef SplitFunctionPixelDifference HDSplitFunctionT;
typedef HyperdepthLeafFunction HDLeafFunctionT;
typedef HyperdepthSplitEvaluator HDSplitEvaluatorT;
typedef Forest<HDSplitFunctionT, HDLeafFunctionT> HDForest;



template <typename T>
class Raw {
public:
  const T* raw;
  const int nsamples;
  const int rows;
  const int cols;
  Raw(const T* raw, int nsamples, int rows, int cols)
    : raw(raw), nsamples(nsamples), rows(rows), cols(cols) {}

  T operator()(int n, int r, int c) const {
    return raw[(n * rows + r) * cols + c];
  }
};


class RawSample : public Sample {
public:
  RawSample(const Raw<uint8_t>& raw, int n, int rc, int cc, int patch_height, int patch_width)
    : Sample(1, patch_height, patch_width), raw(raw), n(n), rc(rc), cc(cc) {}

  virtual float at(int ch, int r, int c) const {
    r += rc - height_ / 2;
    c += cc - width_ / 2;
    r = std::max(0, std::min(raw.rows-1, r));
    c = std::max(0, std::min(raw.cols-1, c));
    return raw(n, r, c);
  }

protected:
  const Raw<uint8_t>& raw;
  int n;
  int rc;
  int cc;
};

void extract_row_samples(const Raw<uint8_t>& im, const Raw<float>& disp, int row, int n_disp_bins, bool only_valid, std::vector<TrainDatum>& data) {
  for(int n = 0; n < im.nsamples; ++n) {
    for(int col = 0; col < im.cols; ++col) {
      float d = disp(n, row, col);
      float pos = col - d;
      int cl = pos * n_disp_bins;
      if((d < 0 || cl < 0) && only_valid) continue;

      auto sample = std::make_shared<RawSample>(im, n, row, col, 32, 32);
      auto target = std::make_shared<ClassificationTarget>(cl);
      auto datum = TrainDatum(sample, target);
      data.push_back(datum);
    }
  }
  std::cout << "extracted " << data.size() << " train samples" << std::endl;
  std::cout << "n_classes (" << im.cols << ") * n_disp_bins (" << n_disp_bins << ") = " << (im.cols * n_disp_bins) << std::endl;
}


void train(int row_from, int row_to, TrainParameters params, const uint8_t* ims, const float* disps, int n, int h, int w, int n_disp_bins, int depth_switch, int n_threads, std::string forest_prefix) {
  Raw<uint8_t> raw_ims(ims, n, h, w);
  Raw<float> raw_disps(disps, n, h, w);

  int n_classes = w;

  auto gen_split_fcn = std::make_shared<HDSplitFunctionT>();
  auto gen_leaf_fcn = std::make_shared<HDLeafFunctionT>(n_classes * n_disp_bins);
  auto split_eval = std::make_shared<HDSplitEvaluatorT>(true, n_classes, n_disp_bins, depth_switch);

  for(int row = row_from; row < row_to; ++row) {
    std::cout << "train row " << row << std::endl;

    std::vector<TrainDatum> data;
    extract_row_samples(raw_ims, raw_disps, row, n_disp_bins, true, data);

    TrainForestQueued<HDSplitFunctionT, HDLeafFunctionT, HDSplitEvaluatorT> train(params, gen_split_fcn, gen_leaf_fcn, split_eval, n_threads, true);

    auto forest = train.Train(data, TrainType::TRAIN, nullptr);

    std::ostringstream forest_path;
    forest_path << forest_prefix << row << ".bin";
    std::cout << "save forest of row " << row << " to " << forest_path.str() << std::endl;
    BinarySerializationOut fout(forest_path.str());
    forest->Save(fout);
  }
}


void eval(int row_from, int row_to, const uint8_t* ims, const float* disps, int n, int h, int w, int n_disp_bins, int depth_switch, int n_threads, std::string forest_prefix, float* out) {
  Raw<uint8_t> raw_ims(ims, n, h, w);
  Raw<float> raw_disps(disps, n, h, w);

  for(int row = row_from; row < row_to; ++row) {
    std::vector<TrainDatum> data;
    extract_row_samples(raw_ims, raw_disps, row, n_disp_bins, false, data);

    std::ostringstream forest_path;
    forest_path << forest_prefix << row << ".bin";
    std::cout << "eval row " << row << " - " << forest_path.str() << std::endl;

    BinarySerializationIn fin(forest_path.str());
    HDForest forest;
    forest.Load(fin);

    auto res = forest.inferencemt(data, n_threads);

    for(int nidx = 0; nidx < n; ++nidx) {
      for(int col = 0; col < w; ++col) {
        auto fcn = res[nidx * w + col];
        int pos, pos2;
        std::tie(pos, pos2) = fcn->argmax();
        float disp = col - float(pos) / n_disp_bins;
        float disp2 = col - float(pos2) / n_disp_bins;

        float prob = fcn->prob_vec()[pos];

        out[((nidx * h + row) * w + col) * 3 + 0] = disp;
        out[((nidx * h + row) * w + col) * 3 + 1] = prob;
        out[((nidx * h + row) * w + col) * 3 + 2] = std::abs(disp - disp2);
      }
    }
  }
}
