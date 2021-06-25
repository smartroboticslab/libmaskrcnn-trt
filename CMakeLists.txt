# SPDX-FileCopyrightText: 2019 NVIDIA Corporation
# SPDX-FileCopyrightText: 2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2021 Sotiris Papatheodorou
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(libmaskrcnn-trt VERSION 0.1.0
	DESCRIPTION "Run Mask R-CNN using TensorRT"
	LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/find_library_create_target.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/set_ifndef.cmake)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++)
endif()

set(CMAKE_SKIP_BUILD_RPATH True)

# Cross Compilation Settings ###################################################
set_ifndef(TRT_PLATFORM_ID ${CMAKE_HOST_SYSTEM_PROCESSOR})
message(STATUS "Target CPU architecture: ${TRT_PLATFORM_ID}")

################################################################################
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")

if (DEFINED GPU_ARCHS)
	message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
	separate_arguments(GPU_ARCHS)
else()
	list(APPEND GPU_ARCHS 35 53 61 70 75)
	string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
	if (IS_ARM)
		# Xavier (SM72) only supported for aarch64.
		list(APPEND GPU_ARCHS 72)
	endif()
	if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
		# Ampere GPU (SM80) support is only available in
		# CUDA versions > 11.0
		list(APPEND GPU_ARCHS 80)
	else()
		message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
	endif()
	message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
set(BERT_GENCODES)
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
	if (${arch} GREATER_EQUAL 70)
		set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
	endif()
	set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
	set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

# Dependencies #################################################################
set(DEFAULT_CUDA_VERSION 10.2)
set(DEFAULT_CUDNN_VERSION 8.1)
set(DEFAULT_PROTOBUF_VERSION 3.0.0)
set(DEFAULT_PROTOBUF_INTERNAL_VERSION 10.0)
set(DEFAULT_CUB_VERSION 1.8.0)

# Dependency Version Resolution
set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")
set_ifndef(CUDNN_VERSION ${DEFAULT_CUDNN_VERSION})
message(STATUS "cuDNN version set to ${CUDNN_VERSION}")
set_ifndef(PROTOBUF_VERSION ${DEFAULT_PROTOBUF_VERSION})
message(STATUS "Protobuf version set to ${PROTOBUF_VERSION}")

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

## find_package(CUDA) is broken for cross-compilation. Enable CUDA language instead.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	find_package(CUDA ${CUDA_VERSION} REQUIRED)
endif()

find_library(CUDNN_LIB cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
# CUBLASLT libraries are only available in CUDA versions > 10. Check for CUDA version here and
# remove dependency on the libarary and unset BERT_GENCODES.
if (CUDA_VERSION VERSION_LESS_EQUAL 10.0)
	message(WARNING "Detected CUDA version is <= 10.0! Removing BERT plugins from compilation list.")
	unset(BERT_GENCODES)
else()
	find_library(CUBLASLT_LIB cublasLt HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
endif()

find_library_create_target(nvinfer nvinfer SHARED ${CMAKE_BINARY_DIR})
find_library_create_target(nvinfer_plugin nvinfer_plugin SHARED ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR})
find_library_create_target(nvuffparser nvparsers SHARED ${CMAKE_BINARY_DIR})
find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)

# libmaskrcnn-trt ##############################################################
set(LIB_NAME maskrcnn-trt)
add_library(${LIB_NAME} STATIC
	src/logger.cpp
	src/maskrcnn_config.cpp
	src/detection.cpp
	src/maskrcnn.cpp
)

target_include_directories(${LIB_NAME}
	PUBLIC
		include
		${CUDA_INCLUDE_DIRS}
		${CUDNN_ROOT_DIR}/include
		${OpenCV_INCLUDE_DIRS}
)

# Necessary to link nvinfer_plugin library.
target_link_libraries(${LIB_NAME}
	${CUDART_LIB}
	${CUBLAS_LIB}
	${CUDNN_LIB}
	nvinfer
	nvinfer_plugin
	nvuffparser
	${OpenCV_LIBS}
	${RT_LIB}
	${CMAKE_DL_LIBS}
	${CMAKE_THREAD_LIBS_INIT}
	stdc++fs
	-Wl,--unresolved-symbols=ignore-in-shared-libs
)

set_target_properties(${LIB_NAME}
	PROPERTIES
		LINK_FLAGS "-Wl,--exclude-libs,ALL"
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Generate pkg-config file #####################################################
configure_file(maskrcnn-trt.pc.in maskrcnn-trt.pc @ONLY)

# Example executable ###########################################################
add_executable(${LIB_NAME}-example src/maskrcnn_example.cpp)
target_link_libraries(${LIB_NAME}-example ${LIB_NAME})

