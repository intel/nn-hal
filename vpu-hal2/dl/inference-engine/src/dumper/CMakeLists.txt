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

set(TARGET_NAME "ieDumper")
if(ENABLE_DUMP)
    add_definitions(-DDEBUG_DUMP)
endif()

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

addVersionDefines(dump_plugin.cpp CI_BUILD_NUMBER)

if(WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)
endif()

include_directories(
        ${IE_MAIN_SOURCE_DIR}/include
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} inference_engine boost_system boost_filesystem)
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
