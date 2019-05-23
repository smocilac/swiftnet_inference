cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)  
set(CMAKE_CXX_FLAGS "-std=c++17 -lstdc++fs")
project (tensorrt_test)

# The version number.
set (tensorrt_test_VERSION_MAJOR 1)
set (tensorrt_test_VERSION_MINOR 0)

# include common headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# tensorrt wrapper project
# set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorRTWrapper/code)
# add_subdirectory(${TRT_WRAPPER_DIR})
# include_directories(${TRT_WRAPPER_DIR}/include)

# setup CUDA
find_package(CUDA)
set(CUDA_VERBOSE_BUILD ON)
message("-- CUDA version: ${CUDA_VERSION}")
if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_61")
	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch=compute_61,code=sm_61
	)
endif()

# opencv
# find_package(OpenCV COMPONENTS core calib3d REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(${OpenCV_LIBRARIES_DIRS})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()


# executable
cuda_add_executable( tensorrt_test main.cpp )
# finally link libraries
cuda_include_directories(${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_include_directories(tensorrt_test PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries( tensorrt_test  ${TENSORRT_LIBRARY} ${CUDA_LIBRARIES} nvinfer nvparsers nvinfer_plugin nvonnxparser )
