#include <iostream>
#include <array>
#include <sstream>
#include <chrono>
#include <memory>

#include "algorithm/FileLoader.h"
#include "algorithm/NNAlgorithm.h"

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

void test(std::shared_ptr<NNAlgorithm> classifier) {
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
		NNAlgorithm::DataType train_data;
		NNAlgorithm::LabelType train_label;
		NNAlgorithm::DataType test_data;
		NNAlgorithm::LabelType test_label;
		NNAlgorithm::LabelType classify_label;

		std::stringstream data_file;
		data_file << "/home/vka/Programming/C/workspace/ecg-classification/data/ReferencyjneDane2/" << file << "/ConvertedQRSRawData_2.txt";
		std::stringstream class_file;
		class_file << "/home/vka/Programming/C/workspace/ecg-classification/data/ReferencyjneDane2/" << file << "/Class_IDs_2.txt";

		FileLoader::load(data_file.str(), class_file.str(),
				train_data, test_data, train_label, test_label, 1, 2.0 / 3);
		auto post_load = high_resolution_clock::now();

		classifier->train(train_data, train_label);

		auto post_train = high_resolution_clock::now();

		classifier->classify(test_data, classify_label);

		auto post_classify = high_resolution_clock::now();

		auto accuracy = classifier->accuracy(test_label, classify_label);

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
}
