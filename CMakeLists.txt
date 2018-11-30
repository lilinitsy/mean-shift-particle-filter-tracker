## TODO: Delete this if for sure going with python

cmake_minimum_required(VERSION 2.8)
project(MeanShift_ParticleTracking)
find_package(OpenCV REQUIRED)

add_executable(MeanShiftTracker src/Mouse.cpp)

target_link_libraries(MeanShiftTracker ${OpenCV_LIBS})
