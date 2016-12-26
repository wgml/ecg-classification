#include <iostream>
#include "algorithm/ENN.h"
#include "tester.h"

int main() {
  const int K = 3;
  std::shared_ptr<NNAlgorithm> classifier = std::make_shared<ENN>(K);
  std::cerr << "Running test suite for ENN with K=" << K << std::endl;
  test(classifier);
  return 0;
}
