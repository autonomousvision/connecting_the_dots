#include "hyperdepth.h"
#include "rf/train.h"

int main() {
  cv::Mat_<uint8_t> im = read_im(0);
  cv::Mat_<uint16_t> disp = read_disp(0);
  int im_rows = im.rows;
  int im_cols = im.cols;
  std::cout << im.rows << "/" << im.cols << std::endl;
  std::cout << disp.rows << "/" << disp.cols << std::endl;

  TrainParameters params;
  params.n_trees = 6;
  params.n_test_samples = 2048;
  params.min_samples_to_split = 16;
  params.min_samples_for_leaf = 8;
  params.n_test_split_functions = 50;
  params.n_test_thresholds = 10;
  params.max_tree_depth = 8;

  int n_classes = im_cols;
  int n_disp_bins = 16;
  int depth_switch = 4;

  auto gen_split_fcn = std::make_shared<HDSplitFunctionT>();
  auto gen_leaf_fcn = std::make_shared<HDLeafFunctionT>(n_classes * n_disp_bins);
  auto split_eval = std::make_shared<HDSplitEvaluatorT>(true, n_classes, n_disp_bins, depth_switch);

  for(int row = 0; row < im_rows; ++row) {
    std::vector<TrainDatum> train_data;
    for(int idx = 0; idx < 12; ++idx) {
      std::cout << "read sample " << idx << std::endl;
      im = read_im(idx);
      disp = read_disp(idx);

      extract_row_samples(im, disp, row, train_data, true, n_disp_bins);
    }
    std::cout << "extracted " << train_data.size() << " train samples" << std::endl;
    std::cout << "n_classes (" << n_classes << ") * n_disp_bins (" << n_disp_bins << ") = " << (n_classes * n_disp_bins) << std::endl;

    TrainForestQueued<HDSplitFunctionT, HDLeafFunctionT, HDSplitEvaluatorT> train(params, gen_split_fcn, gen_leaf_fcn, split_eval, true);

    auto forest = train.Train(train_data, TrainType::TRAIN, nullptr);
    std::cout << "training done" << std::endl;

    std::ostringstream forest_path;
    forest_path << "cforest_" << row << ".bin";
    BinarySerializationOut fout(forest_path.str());
    forest->Save(fout);
  }
}

