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

set(TARGET_NAME "vpu_common")

file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file(GLOB HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

set_source_files_properties(SOURCES PROPERTIES COMPILE_FLAGS -Wall COMPILE_FLAGS -g)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} inference_engine)
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
