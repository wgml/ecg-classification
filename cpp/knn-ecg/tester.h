#include <iostream>
#include <array>
#include <sstream>
#include <chrono>
#include <memory>
#include <vector>
#include <cstring>

#include "algorithm/FileLoader.h"
#include "algorithm/NNAlgorithm.h"

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

namespace {
void do_test(const std::vector<std::string> &directories, std::shared_ptr<NNAlgorithm> classifier,
             bool multi_data_file) {
  const double split_factor = 0.7;

  for (auto &dir : directories) {
    NNAlgorithm::DataType train_data;
    NNAlgorithm::LabelType train_label;
    NNAlgorithm::DataType test_data;
    NNAlgorithm::LabelType test_label;
    NNAlgorithm::LabelType classify_label;

    std::cerr << "Testing " << dir << std::endl;
    auto start_time = high_resolution_clock::now();

    if (multi_data_file) {
      std::stringstream train_data_file;
      std::stringstream test_data_file;
      std::stringstream train_class_file;
      std::stringstream test_class_file;
      train_data_file << dir << "/train_data.txt";
      test_data_file << dir << "/test_data.txt";
      train_class_file << dir << "/train_label.txt";
      test_class_file << dir << "/test_label.txt";
      file_loader::load(train_data_file.str(), test_data_file.str(),
                        train_class_file.str(), test_class_file.str(),
                        train_data, test_data, train_label, test_label,
                        1);
    } else {
      std::stringstream data_file;
      std::stringstream class_file;
      data_file << dir << "/data.txt";
      class_file << dir << "/label.txt";
      file_loader::load(data_file.str(), class_file.str(),
                        train_data, test_data, train_label, test_label,
                        1, split_factor);
    }

    auto post_load = high_resolution_clock::now();
    std::cerr << "There are " << train_data.rows() << " samples in train vector and "
              << test_data.rows() << " samples in test vector." << std::endl;
    classifier->train(train_data, train_label);

    auto post_train = high_resolution_clock::now();

    classifier->classify(test_data, classify_label);

    auto post_classify = high_resolution_clock::now();

    auto accuracy = classifier->accuracy(test_label, classify_label);

    auto total_time = duration_cast<milliseconds>(post_classify - start_time).count();
    auto load_time = duration_cast<milliseconds>(post_load - start_time).count();
    auto train_time = duration_cast<milliseconds>(post_train - post_load).count();
    auto classify_time = duration_cast<milliseconds>(post_classify - post_train).count();

    std::cerr << "Accuracy for dir '" << dir << "' is " << accuracy << "%." << std::endl;
    if (multi_data_file)
      std::cerr << "Classification was based on split train and test files. ";
    else
      std::cerr << "Classification was based data split with " << split_factor << " factor. ";
    std::cerr << "It took me " << total_time << "ms ("
              << load_time << "ms for loading data, "
              << train_time << "ms for training and "
              << classify_time << "ms for classification)." << std::endl;
  }
  std::cerr << "Bye" << std::endl;
}

void do_test_time(std::shared_ptr<NNAlgorithm> classifier, NNAlgorithm::DataType train_data,
                  NNAlgorithm::DataType test_data, NNAlgorithm::LabelType train_label) {
  auto start = high_resolution_clock::now();
  NNAlgorithm::LabelType result;
  classifier->train(train_data, train_label);
  classifier->classify(test_data, result);

  auto train_data_amount = train_data.rows();
  auto test_data_amount = test_data.rows();
  long time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
  std::cout << train_data_amount << " " << test_data_amount << " " << time << std::endl;
}

void test_time(const std::vector<std::string> &directories, std::shared_ptr<NNAlgorithm> classifier,
               bool multi_data_file) {
  const double split_factor = 0.7;

  for (auto &dir : directories) {
    NNAlgorithm::DataType train_data_input;
    NNAlgorithm::LabelType train_label_input;
    NNAlgorithm::DataType test_data_input;
    NNAlgorithm::LabelType test_label_input;
    NNAlgorithm::LabelType classify_label;

    std::cerr << "Testing " << dir << std::endl;

    if (multi_data_file) {
      std::stringstream train_data_file;
      std::stringstream test_data_file;
      std::stringstream train_class_file;
      std::stringstream test_class_file;
      train_data_file << dir << "/train_data_input.txt";
      test_data_file << dir << "/test_data.txt";
      train_class_file << dir << "/train_label.txt";
      test_class_file << dir << "/test_label.txt";
      file_loader::load(train_data_file.str(), test_data_file.str(),
                        train_class_file.str(), test_class_file.str(),
                        train_data_input, test_data_input, train_label_input, test_label_input,
                        1);
    } else {
      std::stringstream data_file;
      std::stringstream class_file;
      data_file << dir << "/data.txt";
      class_file << dir << "/label.txt";
      file_loader::load(data_file.str(), class_file.str(),
                        train_data_input, test_data_input, train_label_input, test_label_input,
                        1, split_factor);
    }

    long samples_num_thousands = std::min(train_label_input.rows(), test_label_input.rows()) / 1000;
    long samples_num = 1000 * samples_num_thousands;
    auto train_data = train_data_input.block(0, 0, samples_num, train_data_input.cols());
    auto test_data = test_data_input.block(0, 0, samples_num, test_data_input.cols());
    auto train_label = train_label_input.block(0, 0, samples_num, train_label_input.cols());

    for (int samples = 1000; samples <= samples_num; samples += 1000) {
      auto train_data_cut = train_data.block(0, 0, samples, train_data.cols());
      auto test_data_cut = test_data.block(0, 0, samples, test_data.cols());
      auto train_label_cut = train_label.block(0, 0, samples, train_label.cols());

      if (samples < samples_num) {
        do_test_time(classifier, train_data_cut, test_data_cut, train_label_cut);
        do_test_time(classifier, train_data_cut, test_data, train_label_cut);
      }
      do_test_time(classifier, train_data, test_data_cut, train_label);
    }
  }
  std::cerr << "Bye" << std::endl;
}

std::vector<std::string> directories(int len, char *argv[]) {
  std::vector<std::string> dirs{static_cast<unsigned int>(len)};
  for (auto i = 0; i < len; i++)
    dirs[i] = argv[i];
  return dirs;
}
}

namespace testing {
void test(std::shared_ptr<NNAlgorithm> classifier, int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Pass testing directories as command line arguments." << std::endl;
  } else {
    bool multi_data_file = false;
    bool speed_test = false;

    int from = 1;
    while (true) {
      if (strcmp("--multi-data-file", argv[from]) == 0) {
        multi_data_file = true;
        from++;
      } else if (strcmp("--speed-test", argv[from]) == 0) {
        speed_test = true;
        from++;
      } else
      break;
    }
    if (speed_test)
      test_time(directories(argc - from, argv + from), classifier, multi_data_file);
    else
      do_test(directories(argc - from, argv + from), classifier, multi_data_file);
  }
}
}
