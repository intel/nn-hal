# Copyright (c) 2017 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(TARGET_NAME "HDDLPlugin")

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

addVersionDefines(hddl_plugin.cpp CI_BUILD_NUMBER)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${HDDL}/include
)

if (WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)
endif ()

find_library(HDDLAPI_LIBRARY
        "hddlapi"
        PATHS "${HDDL}/lib"
        NO_DEFAULT_PATH)

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME}  inference_engine ${INTEL_ITT_LIBS} graph_transformer vpu_common ${HDDLAPI_LIBRARY})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# saving rpath to be used by CI
log_rpath(HDDL ${HDDLAPI_LIBRARY})
