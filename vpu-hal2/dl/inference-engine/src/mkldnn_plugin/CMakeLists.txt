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

set(TARGET_NAME "MKLDNNPlugin")

if (UNIX AND NOT APPLE)
    set(OS_SPECIFIC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/os/lin/*.cpp)
endif()

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp
        ${OS_SPECIFIC_SRC}
)

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.h
)

addVersionDefines(mkldnn_plugin.cpp CI_BUILD_NUMBER MKL_VERSION)

if(WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)
endif()

include_directories(
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include
)

enable_omp()

if (ENABLE_MKL_DNN_GEMM)
    log_rpath_from_dir(MKL "${MKL}/lib")
endif()

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} inference_engine ${INTEL_ITT_LIBS} mkldnn "${intel_omp_lib}")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

add_library(test_${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})

target_link_libraries(test_${TARGET_NAME} inference_engine_s mkldnn "${intel_omp_lib}")
set_target_properties(test_${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME test_${TARGET_NAME})
