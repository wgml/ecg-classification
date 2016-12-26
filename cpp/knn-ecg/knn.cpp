#include <iostream>
#include "algorithm/KNN.h"
#include "tester.h"

int main() {
  const int K = 3;
  std::shared_ptr<NNAlgorithm> classifier = std::make_shared<KNN>(K);
  std::cerr << "Running test suite for KNN with K=" << K << std::endl;
  test(classifier);
  return 0;
}
