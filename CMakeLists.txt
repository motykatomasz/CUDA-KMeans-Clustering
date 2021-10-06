cmake_minimum_required(VERSION 3.10)

project(kmeans)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
set(CMAKE_BUILD_TYPE Debug)

set(DEFAULT_SOURCES
        src/main.cpp
        src/utils/RandomGenerator.hpp
        src/utils/Timer.hpp
        src/utils/FeatureVec.cpp src/utils/FeatureVec.hpp
        src/utils/DataSet.cpp src/utils/DataSet.hpp
        src/IKMeans.hpp
        src/cpu/KMeans.cpp src/cpu/KMeans.hpp
        src/cuda/KMeansCuda.cpp src/cuda/KMeansCuda.hpp)



include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
        message("CUDA is supported")
        enable_language(CUDA)
        add_definitions(-DUSE_CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CUDA_SOURCES
                src/cuda/Utils.cu
                src/cuda/Kernels.cu
                src/cuda/CudaErrors.hpp)

else ()
        message("Could not find CUDA support.")
endif ()

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(${PROJECT_NAME} ${DEFAULT_SOURCES} ${CUDA_SOURCES})

# For cuda-gdb
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g -G >)
