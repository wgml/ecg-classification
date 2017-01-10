#include "ENN.h"
#include <igl/unique.h>
#include <igl/slice.h>

void ENN::train(const ENN::DataType &train_data, const ENN::LabelType &train_labels) {
  assert(train_data.rows() == train_labels.rows());

  this->train_data = train_data;
  this->train_labels = train_labels;

  LabelType unique_labels_sorted;
  unique_labels = LabelType{};
  igl::unique(train_labels, unique_labels);

  const auto uniq_labels_num = unique_labels.rows();
  const auto train_amount = train_labels.rows();
  {
    IndexType uniq_classes_indexes{uniq_labels_num};
    igl::sort(unique_labels, 1, true, unique_labels_sorted, uniq_classes_indexes);
  }

  nn_distances = Eigen::MatrixXd{train_amount, K};
  nn_labels = NNLabelsType{train_amount, K};

  #pragma omp parallel for
  for (int i = 0; i < train_amount; i++) {
    auto train_sample = train_data.row(i);
    DistanceType distances{train_amount}, sorted_distances{train_amount};
    IndexType distance_indexes{train_amount};
    distance(train_data, train_sample, distances);
    igl::sort(distances, 1, true, sorted_distances, distance_indexes);
    nn_distances.row(i) = sorted_distances.block(1, 0, K, 1).transpose();

    for (unsigned int k = 0; k < K; k++)
      nn_labels(i, k) = train_labels(distance_indexes(k + 1));
  }

  n_i = Eigen::VectorXi{uniq_labels_num, 1};
  T = Eigen::VectorXd{uniq_labels_num, 1};

  for (unsigned int label_id = 0; label_id < uniq_labels_num; label_id++) {
    auto label = unique_labels(label_id);
    unsigned int n_i_val = 0, T_val = 0;
    for (unsigned int train_sample_id = 0; train_sample_id < train_amount; train_sample_id++) {
      if (train_labels(train_sample_id) == label) {
        n_i_val++;
        for (unsigned int k = 0; k < K; k++) {
          if (nn_labels(train_sample_id, k) == label)
            T_val++;
        }
      }
    }
    n_i(label_id) = n_i_val;
    T(label_id) = (1.0 * T_val) / (n_i_val * K);
  }
}

void ENN::classify(const ENN::DataType &test_data, ENN::LabelType &result) const {
  auto test_amount = test_data.rows();
  auto train_amount = train_data.rows();
  const auto uniq_labels_num = unique_labels.rows();

  if (test_amount == 0)
    return;
  result.resize(test_amount, Eigen::NoChange);

  #pragma omp parallel for
  for (unsigned int test_sample_id = 0; test_sample_id < test_amount; test_sample_id++) {
    const auto test_sample = test_data.row(test_sample_id);
    DistanceType distances{train_amount};
    distance(train_data, test_sample, distances);

    IndexType distance_indexes{train_amount};
    {
      DistanceType distances_sorted{train_amount};
      igl::sort(distances, 1, true, distances_sorted, distance_indexes);
    }
    LabelType x_nn_labels{K};
    for (unsigned int k = 0; k < K; k++) {
      x_nn_labels(k) = train_labels(distance_indexes(k));
    }
    Eigen::VectorXi k_i{uniq_labels_num};

    for (unsigned int i = 0; i < uniq_labels_num; i++) {
      unsigned int k_i_val = 0;
      for (unsigned int k = 0; k < K; k++) {
        if (x_nn_labels(k) == unique_labels(i))
          k_i_val++;
      }
      k_i(i) = k_i_val;
    }
    Eigen::VectorXd predictions{uniq_labels_num};
    Eigen::VectorXi labels_num_ij{uniq_labels_num};
    Eigen::VectorXi labels_num_jj{uniq_labels_num};

    for (unsigned int label_idx = 0; label_idx < uniq_labels_num; label_idx++) {
      const auto label = unique_labels(label_idx);
      unsigned int labels_num_ij_val = 0, labels_num_jj_val = 0;
      for (unsigned int sample_idx = 0; sample_idx < train_amount; sample_idx++) {
        if (train_labels(sample_idx) == label) {
          const auto orig_nn_distance = nn_distances(sample_idx, K - 1);
          const auto orig_nn_label = nn_labels(sample_idx, K - 1);

          const auto diff_distance = distances(sample_idx) - orig_nn_distance;
          if (diff_distance < 0) {
            labels_num_ij_val++;
            if (orig_nn_label == label)
              labels_num_jj_val++;
          }
        }
      }
      labels_num_ij(label_idx) = labels_num_ij_val;
      labels_num_jj(label_idx) = labels_num_jj_val;
    }
    for (unsigned int label_idx = 0; label_idx < uniq_labels_num; label_idx++) {
      auto delta_n_jj = labels_num_ij(label_idx) - labels_num_jj(label_idx);
      double s1 = (delta_n_jj + k_i(label_idx) - T(label_idx) * K) / ((n_i(label_idx) + 1) * K);
      double s2 = 0;
      for (unsigned int i = 0; i < uniq_labels_num; i++)
        s2 += (1.0 * labels_num_jj(i)) / (n_i(i) * K);
      s2 -= (1.0 * labels_num_jj(label_idx)) / (n_i(label_idx) * K);
      predictions(label_idx) = s1 - s2;
    }
    double max_pred = NAN;
    ClassType selected_label{0};
    for (unsigned int i = 0; i < uniq_labels_num; i++) {
      if (std::isnan(max_pred) || predictions(i) > max_pred) {
        max_pred = predictions(i);
        selected_label = unique_labels(i);
      }
    }
    result(test_sample_id) = selected_label;
  }
}
