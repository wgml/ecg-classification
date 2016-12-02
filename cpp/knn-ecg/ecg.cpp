// tip: http://igl.ethz.ch/projects/libigl/matlab-to-eigen.html

#include <iostream>
#include "algorithm/FileLoader.h"
#include "algorithm/KNN.h"
#include "algorithm/ENN.h"
#include <array>
#include <sstream>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

int main() {
	std::array<std::string, 38> files{ {
		"100", "101", "102", "103", "104",
		"105", "106", "108", "109", "111",
		"112", "113", "118", "119", "121",
		"122", "124", "200", "201", "202",
		"203", "205", "208", "209", "210",
		"212", "213", "214", "215", "217",
		"219", "221", "222", "223", "228",
		"231", "233", "234"
	} };

	for (auto &file : files) {
		auto start_time = high_resolution_clock::now();
		const auto K = 3;
//		KNN knn_classifier{3};
		ENN enn_classifier{K};
		KNN::DataType train_data;
		KNN::LabelType train_label;
		KNN::DataType test_data;
		KNN::LabelType test_label;
		KNN::LabelType classify_label;

		std::stringstream data_file;
		data_file << "/home/vka/Programming/C/workspace/ecg-classification/ReferencyjneDane2/" << file << "/ConvertedQRSRawData_2.txt";
		std::stringstream class_file;
		class_file << "/home/vka/Programming/C/workspace/ecg-classification/ReferencyjneDane2/" << file << "/Class_IDs_2.txt";

		FileLoader::load(data_file.str(), class_file.str(),
				train_data, test_data, train_label, test_label, 1, 2.0 / 3);
		auto post_load = high_resolution_clock::now();

		enn_classifier.train(train_data, train_label);

		auto post_train = high_resolution_clock::now();

		enn_classifier.classify(test_data, classify_label);

		auto post_classify = high_resolution_clock::now();

		auto accuracy = enn_classifier.accuracy(test_label, classify_label);

		auto total_time = duration_cast<milliseconds>(post_classify - start_time).count();
		auto load_time = duration_cast<milliseconds>(post_load - start_time).count();
		auto train_time = duration_cast<milliseconds>(post_train - post_load).count();
		auto classify_time = duration_cast<milliseconds>(post_classify - post_train).count();

		std::cerr << "Accuracy for file '" << file << "' is " << accuracy << "%." << std::endl
				<< "It took me " << total_time << "ms ("
				<< load_time << "ms for loading data, "
				<< train_time << "ms for training and "
				<< classify_time << "ms for classification)." << std::endl;
	}
	std::cerr << "Bye" << std::endl;
	return 0;
}
