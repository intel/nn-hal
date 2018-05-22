#===============================================================================
# Copyright (c) 2016 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
#
#  Brief description: This cmake file replase original mkl-dnn build scripts
#  for more convenient integration to IE build process
#
#===============================================================================

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET mkldnn)
set(MKLDNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mkl-dnn)

file(GLOB_RECURSE HDR
        ${MKLDNN_ROOT}/include/*.h
        ${MKLDNN_ROOT}/include/*.hpp
)
file(GLOB_RECURSE SRC
        ${MKLDNN_ROOT}/src/*.c
        ${MKLDNN_ROOT}/src/*.cpp
        ${MKLDNN_ROOT}/src/*.h
        ${MKLDNN_ROOT}/src/*.hpp
)
include_directories(
        ${MKLDNN_ROOT}/include
        ${MKLDNN_ROOT}/src
        ${MKLDNN_ROOT}/src/common
        ${MKLDNN_ROOT}/src/cpu/xbyak
)

if(WIN32)
    add_definitions(-D_WIN)
    add_definitions(-DNOMINMAX)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qlong-double")
    endif()
endif()

enable_omp()

## enable cblas_gemm from mlkml package
set(MKLROOT ${MKL})
include(mkl-dnn/cmake/MKL.cmake)

add_library(${TARGET} STATIC ${HDR} ${SRC})
target_link_libraries(${TARGET} ${${TARGET}_LINKER_LIBS})
