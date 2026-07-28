cmake_minimum_required(VERSION 3.20)
project(fast_scale LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_scale scale.cpp)
target_compile_features(fast_scale PRIVATE cxx_std_17)
