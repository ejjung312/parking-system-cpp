option(OPENCL_ACC "Enable OpenCL acceleration" OFF)
option(CUDA_ACC "Enable CUDA acceleration" OFF)

cmake_minimum_required(VERSION 3.10)
project(parking-system)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG_MODE)
endif()

if (CUDA_ACC)
    message(STATUS "CUDA Acceleration: ${CUDA_ACC}")
    add_definitions(-DCUDA_ACC)
endif()

if (OPENCL_ACC)
    message(STATUS "OpenCL Acceleration: ${OPENCL_ACC}")
    add_definitions(-DOPENCL_ACC)
endif()

# Find OpenCV
if (WIN32) 
    set(OpenCV_DIR "C:\\opencv\\build\\x64\\vc16\\lib")
else()
    set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
endif()
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Source files
set(SOURCES
    src/yolo11.cpp
    src/main.cpp
)

# Executable
add_executable(parking-system ${SOURCES})

# Link OpenCV libraries
target_link_libraries(parking-system ${OpenCV_LIBS})
