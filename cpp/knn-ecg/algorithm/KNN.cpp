#include "KNN.h"

#include <igl/sort.h>

void KNN::train(const KNN::DataType &train_data, const KNN::LabelType &train_labels) {
  this->train_data = train_data;
  this->train_labels = train_labels;
}

void KNN::classify(KNN::DataType &test_data, KNN::LabelType &result) const {
  long test_amount = test_data.rows();
  if (test_amount == 0)
    return;
  result.resize(test_amount, Eigen::NoChange);


  #pragma omp parallel for
  for (int sample = 0; sample < test_amount; sample++) {
    const auto &test_sample = test_data.row(sample);
    DistanceType distances{train_data.rows()};
    distance(train_data, test_sample, distances);
    result(sample) = get_mode_from_k_neighbours(distances);
  }
}

KNN::ClassType KNN::get_mode_from_k_neighbours(KNN::DistanceType &distances) const {

  const auto samples = distances.rows();
  IndexType indexes{samples};
  DistanceType distances_sorted{samples};
  igl::sort(distances, 1, true, distances_sorted, indexes);
  IndexType k_indexes = indexes.block(0, 0, K, 1);
  LabelType k_classes{K};

  for (unsigned int i = 0; i < K; i++)
    k_classes(i) = train_labels(k_indexes(i));
  return mode(k_classes);
}
