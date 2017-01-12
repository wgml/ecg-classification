#ifndef ALGORITHM_KNN_H_
#define ALGORITHM_KNN_H_

#include "NNAlgorithm.h"

struct KNN : public NNAlgorithm {

  KNN(unsigned int K)
      : NNAlgorithm(K) {}

  virtual void train(const DataType &train_data, const LabelType &train_labels) override;

  virtual void classify(DataType &test_data, LabelType &result) const override;

  virtual ~KNN() = default;

private:
  ClassType get_mode_from_k_neighbours(DistanceType &distances) const;

  DataType train_data;
  LabelType train_labels;
};

#endif /* ALGORITHM_KNN_H_ */
