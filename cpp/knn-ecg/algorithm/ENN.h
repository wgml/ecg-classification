#ifndef ALGORITHM_ENN_H_
#define ALGORITHM_ENN_H_

#include <exception>

#include "NNAlgorithm.h"

struct ENN : public NNAlgorithm {

	ENN(const size_t K)
	    : K(K)
	{}

	virtual void train(const DataType &train_data, const LabelType &train_labels) = 0; // todo

	virtual void classify(const DataType &test_data, LabelType &result) = 0; // todo

	virtual ~ENN() = default;

private:
	const size_t K;
};


#endif /* ALGORITHM_ENN_H_ */
