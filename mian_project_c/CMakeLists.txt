cmake_minimum_required(VERSION 3.5)

project(chess)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(chess main.cpp)

target_link_libraries(chess ${OpenCV_LIBS})