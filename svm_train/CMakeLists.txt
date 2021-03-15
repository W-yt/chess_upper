cmake_minimum_required(VERSION 3.5)

project(Chess_svm)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Chess_svm  main.cpp)

target_link_libraries(Chess_svm ${OpenCV_LIBS})