cmake_minimum_required(VERSION 3.0)
project(knn_ecg)

set(CMAKE_CXX_FLAGS "-O3 -pedantic -Werror -Wall -Wextra -std=c++11 -fopenmp -DNDEBUG")

set(EIGEN_INCLUDE "${EIGEN3_INCLUDE_DIR}")
if (NOT EIGEN_INCLUDE)
    message("Eigen include dir not specified via EIGEN3_INCLUDE_DIR. Assuming submodule is being used.")
    set(EIGEN_INCLUDE "deps/eigen")
endif()
set(IGL_INCLUDE "${IGL_INCLUDE}")
if (NOT IGL_INCLUDE)
    message("Igl include dir not specified via IGL_INCLUDE. Assuming submodule is being used.")
    set(IGL_INCLUDE "deps/libigl/include")
endif()
include_directories(SYSTEM "${EIGEN_INCLUDE}" "${IGL_INCLUDE}")

set(SOURCE_FILES
    algorithm/ENN.cpp
    algorithm/ENN.h
    algorithm/FileLoader.h
    algorithm/KNN.cpp
    algorithm/KNN.h
    algorithm/NNAlgorithm.h
    tester.h)

add_library(classifiers SHARED ${SOURCE_FILES})
target_link_libraries(classifiers pthread)

add_executable(knn knn.cpp)
add_executable(enn enn.cpp)

target_link_libraries(knn classifiers)
target_link_libraries(enn classifiers)

