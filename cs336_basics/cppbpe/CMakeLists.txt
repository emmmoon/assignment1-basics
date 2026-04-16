cmake_minimum_required(VERSION 3.15...3.27)
project(c_bpe LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(c_bpe c_bpe.cpp)
set_target_properties(c_bpe PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

install(TARGETS c_bpe DESTINATION cs336_basics)