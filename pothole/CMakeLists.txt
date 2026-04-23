cmake_minimum_required(VERSION 3.10)
project(pothole_detection)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.10 REQUIRED COMPONENTS
    common io filters segmentation search
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pothole_detection pothole_detection.cpp)
target_link_libraries(pothole_detection ${PCL_LIBRARIES})
