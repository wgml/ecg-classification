#ifndef ALGORITHM_ENN_H_
#define ALGORITHM_ENN_H_

#include "NNAlgorithm.h"

struct ENN : public NNAlgorithm {

	ENN(unsigned int K)
	: NNAlgorithm(K)
	{}

	virtual void train(const DataType &train_data, const LabelType &train_labels) override;

	virtual void classify(const DataType &test_data, LabelType &result) const override;

	virtual ~ENN() = default;

private:
	using NNLabelsType = Eigen::Matrix<ClassType, Eigen::Dynamic, Eigen::Dynamic>;

	DataType train_data;
	LabelType train_labels;
	LabelType unique_labels;

	Eigen::MatrixXd nn_distances;
	NNLabelsType nn_labels;
	Eigen::VectorXi n_i;
	Eigen::VectorXd T;
};


#endif /* ALGORITHM_ENN_H_ */
