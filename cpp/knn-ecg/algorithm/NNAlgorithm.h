#ifndef ALGORITHM_NNALGORITHM_H_
#define ALGORITHM_NNALGORITHM_H_

#include <Eigen/Dense>
#include <utility>
#include <cassert>
#include <unordered_map>

struct NNAlgorithm {
	using ClassType = unsigned int;
	using DataType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>; //todo rowmajor vs colmajor (row is a must if using igl)
	using LabelType = Eigen::Matrix<ClassType, Eigen::Dynamic, 1>;
	using DistanceType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
	using IndexType = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

	NNAlgorithm(unsigned int K)
	: K(K)
	{}

	virtual void train(const DataType &train_data, const LabelType &train_labels) = 0;
	virtual void classify(const DataType &test_data, LabelType &result) = 0;

	double accuracy(const LabelType &expected, const LabelType &actual) {
		assert(expected.rows() == actual.rows());
		int samples = static_cast<int>(expected.rows());
		int matched = 0;

		for (int i = 0; i < samples; i++) {
			if (expected(i, 0) == actual(i, 0))
				matched++;
		}
		return (100.0 * matched) / samples;
	}

	void distance(const DataType &train_data, const DataType &test_sample, DistanceType &distances) {
		assert(test_sample.rows() == 1);
		assert(train_data.rows() == distances.rows());
		assert(train_data.cols() == test_sample.cols());
		distances = (DataType::Ones(train_data.rows(), 1) * test_sample - train_data).array().square().rowwise().sum().sqrt().matrix();
	}

	ClassType mode(const LabelType &labels) {
		assert(labels.rows() > 0);

		std::unordered_map<ClassType, int> occurences;
		ClassType top_element = ClassType{0};
		int top_value = 0;

		for (int i = 0; i < labels.rows(); i++) {
			ClassType label = labels(i);
			auto o = ++(occurences[label]);
			if (o > top_value || (o == top_value && top_element > label)) { // matlab-compatible communist mode (favors lower class)
				top_value = o;
				top_element = label;
			}
		}
		return top_element;
	}

	virtual ~NNAlgorithm() = default;

protected:
	const size_t K;
};

#endif /* ALGORITHM_NNALGORITHM_H_ */
