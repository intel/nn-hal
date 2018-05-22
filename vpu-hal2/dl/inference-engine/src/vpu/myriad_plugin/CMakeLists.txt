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

set(TARGET_NAME "myriadPlugin")

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

addVersionDefines(myriad_plugin.cpp CI_BUILD_NUMBER)

set_source_files_properties(SOURCES PROPERTIES COMPILE_FLAGS -Wall COMPILE_FLAGS -g)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MYRIAD}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if (WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN)
endif ()

add_library(libmvnc STATIC IMPORTED)
SET_TARGET_PROPERTIES(libmvnc PROPERTIES IMPORTED_LOCATION "${MYRIAD}/lib/libmvnc.a")

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} inference_engine ${INTEL_ITT_LIBS} graph_transformer vpu_common libmvnc usb-1.0)
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# copy firmware to the directory with binaries
add_custom_command(TARGET ${TARGET_NAME}
          POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib/mvnc
          COMMAND ${CMAKE_COMMAND} -E copy_directory ${MYRIAD}/lib/mvnc/ ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib/mvnc
)
