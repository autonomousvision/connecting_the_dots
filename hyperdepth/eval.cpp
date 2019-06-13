#include "hyperdepth.h"


int main() {
  cv::Mat_<uint8_t> im = read_im(0);
  cv::Mat_<uint16_t> disp = read_disp(0);
  int im_rows = im.rows;
  int im_cols = im.cols;
  std::cout << im.rows << "/" << im.cols << std::endl;
  std::cout << disp.rows << "/" << disp.cols << std::endl;

  cv::Mat_<uint16_t> ta_disp(im_rows, im_cols);
  cv::Mat_<uint16_t> es_disp(im_rows, im_cols);

  int n_disp_bins = 16;

  for(int row = 0; row < im_rows; ++row) {
    std::vector<TrainDatum> data;
    extract_row_samples(im, disp, row, data, false, n_disp_bins);

    std::ostringstream forest_path;
    forest_path << "cforest_" << row << ".bin";
    BinarySerializationIn fin(forest_path.str());
    HDForest forest;
    forest.Load(fin);

    auto res = forest.inferencemt(data, 18);
    for(int col = 0; col < im_cols; ++col) {
      auto fcn = res[col];
      auto target = std::static_pointer_cast<ClassificationTarget>(data[col].target);

      float ta = col - float(target->cl()) / n_disp_bins;
      float es = col - float(fcn->argmax()) / n_disp_bins;
      es = std::max(0.f, es);

      ta_disp(row, col) = int(ta * 16);
      es_disp(row, col) = int(es * 16);
    }
  }

  cv::imwrite("disp_orig.png", disp);
  cv::imwrite("disp_ta.png", ta_disp);
  cv::imwrite("disp_es.png", es_disp);
}


