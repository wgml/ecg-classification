#ifndef ALGORITHM_FILELOADER_H_
#define ALGORITHM_FILELOADER_H_

#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <iterator>
#include <cmath>
#include <Eigen/Dense>

#include "NNAlgorithm.h"

namespace {
void load_data(const std::string &filename, NNAlgorithm::DataType &output,
               const int skip_first_n_cols) {
  std::ifstream input_stream{filename};
  assert(input_stream.is_open());

  std::string line;
  std::vector<std::vector<double>> file_content;

  int known_width = -1;
  while (std::getline(input_stream, line)) {
    std::stringstream stream{line};
    std::istream_iterator<double> start(stream), end;
    std::advance(start, skip_first_n_cols);

    std::vector<double> line_content(start, end);

    if (known_width < 0)
      known_width = static_cast<int>(line_content.size());
    else
      assert(known_width == static_cast<int>(line_content.size()));

    file_content.push_back(line_content);
  }

  output.resize(file_content.size(), known_width);
  int row = 0;
  int col = 0;
  for (auto &line_content : file_content) {
    for (auto &element : line_content)
      output(row, col++) = element;
    row++;
    col = 0;
  }
}

void load_label(const std::string &filename, NNAlgorithm::LabelType &output) {
  std::ifstream input_stream{filename};
  assert(input_stream.is_open());

  std::istream_iterator<NNAlgorithm::ClassType> start(input_stream), end;
  std::vector<NNAlgorithm::ClassType> file_content(start, end);

  output.resize(file_content.size(), 1);
  int i = 0;
  for (auto &element : file_content)
    output(i++) = element;
}

void normalize(NNAlgorithm::DataType &data) {
  const unsigned int stdev_degrees_of_freedom = 1; // matlab-compatible result
  for (auto col = 0; col < data.cols(); col++) {
    auto data_column = data.col(col);
    const auto mean = data_column.mean();
    data_column.array() -= mean;
    const auto stdev = std::sqrt(
        (data_column.array()).square().sum() / (data_column.rows() - stdev_degrees_of_freedom));
    data_column.array() /= stdev;
  }
}

void normalize(NNAlgorithm::DataType &first, NNAlgorithm::DataType &second) {
  const unsigned int stdev_degrees_of_freedom = 1; // matlab-compatible result
  auto cols = first.cols();
  for (auto col = 0; col < cols; col++) {
    auto first_column = first.col(col);
    auto second_column = second.col(col);
    NNAlgorithm::DataType joined(first_column.rows() + second_column.rows(), 1);
    joined << first_column, second_column;

    const auto mean = first_column.mean();
    first_column.array() -= mean;
    second_column.array() -= mean;

    joined << first_column, second_column;

    const auto stdev = std::sqrt((joined.array()).square().sum()
                                 / (joined.rows() - stdev_degrees_of_freedom));
    first_column.array() /= stdev;
    second_column.array() /= stdev;
  }
}

void do_split(NNAlgorithm::DataType &data, NNAlgorithm::LabelType &label,
              long samples1, long samples2,
              NNAlgorithm::DataType &data_out1, NNAlgorithm::DataType &data_out2,
              NNAlgorithm::LabelType &label_out1, NNAlgorithm::LabelType &label_out2) {
  int sample_size = static_cast<int>(data.cols());
  data_out1.resize(samples1, sample_size);
  data_out1 << data.block(0, 0, samples1, sample_size);

  data_out2.resize(samples2, sample_size);
  data_out2 << data.block(samples1, 0, samples2, sample_size);

  label_out1.resize(samples1, 1);
  label_out1 << label.block(0, 0, samples1, 1);

  label_out2.resize(samples2, 1);
  label_out2 << label.block(samples1, 0, samples2, 1);
}

void split_with_ratio(NNAlgorithm::DataType &data, NNAlgorithm::LabelType &label,
                      double split_ratio,
                      NNAlgorithm::DataType &data_out1, NNAlgorithm::DataType &data_out2,
                      NNAlgorithm::LabelType &label_out1, NNAlgorithm::LabelType &label_out2) {
  assert(split_ratio >= 0 && split_ratio <= 1);
  assert(data.rows() == label.rows());
  long samples = static_cast<long>(data.rows());
  long samples1 = static_cast<long>(samples * split_ratio);
  long samples2 = samples - samples1;
  do_split(data, label, samples1, samples2, data_out1, data_out2, label_out1, label_out2);
}

void split_with_value(NNAlgorithm::DataType &data, NNAlgorithm::LabelType &label,
                      long train_samples,
                      NNAlgorithm::DataType &data_out1, NNAlgorithm::DataType &data_out2,
                      NNAlgorithm::LabelType &label_out1, NNAlgorithm::LabelType &label_out2) {
  assert(train_samples >= 0 && train_samples <= data.rows());
  assert(data.rows() == label.rows());
  long samples1 = train_samples;
  long samples2 = static_cast<long>(data.rows()) - samples1;
  do_split(data, label, samples1, samples2, data_out1, data_out2, label_out1, label_out2);
}
}

namespace file_loader {
void load(const std::string &data_file, const std::string &label_file,
          NNAlgorithm::DataType &train_data, NNAlgorithm::DataType &test_data,
          NNAlgorithm::LabelType &train_label, NNAlgorithm::LabelType &test_label,
          const int skip_first_n_cols, const double split_ratio) {
  NNAlgorithm::DataType data_tmp;
  NNAlgorithm::LabelType label_tmp;
  load_data(data_file, data_tmp, skip_first_n_cols);
  load_label(label_file, label_tmp);
  normalize(data_tmp);
  split_with_ratio(data_tmp, label_tmp, split_ratio,
                   train_data, test_data, train_label, test_label);
}

void load(const std::string &train_data_file, const std::string &test_data_file,
          const std::string &train_label_file, const std::string &test_label_file,
          NNAlgorithm::DataType &train_data, NNAlgorithm::DataType &test_data,
          NNAlgorithm::LabelType &train_label, NNAlgorithm::LabelType &test_label,
          const int skip_first_n_cols) {
  // unimaginably ugly but somehow faster than filling output containers directly

  NNAlgorithm::DataType data_tmp1, data_tmp2, data_concat;
  NNAlgorithm::LabelType label_tmp1, label_tmp2, label_concat;

  load_data(train_data_file, data_tmp1, skip_first_n_cols);
  load_data(test_data_file, data_tmp2, skip_first_n_cols);
  data_concat.resize(data_tmp1.rows() + data_tmp2.rows(), data_tmp1.cols());
  data_concat << data_tmp1, data_tmp2;

  load_label(train_label_file, label_tmp1);
  load_label(test_label_file, label_tmp2);
  label_concat.resize(label_tmp1.rows() + label_tmp2.rows(), label_tmp1.cols());
  label_concat << label_tmp1, label_tmp2;

  normalize(train_data, test_data);
  split_with_value(data_concat, label_concat, label_tmp1.rows(),
                   train_data, test_data, train_label, test_label);

  assert(label_tmp1.rows() == train_label.rows());
}
}

#endif /* ALGORITHM_FILELOADER_H_ */
