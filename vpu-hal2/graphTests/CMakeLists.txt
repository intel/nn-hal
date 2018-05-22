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
cmake_minimum_required(VERSION 2.8)

set (TARGET_NAME "graphTests")

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

# Properties->C/C++->General->Additional Include Directories
include_directories (
		${CMAKE_CURRENT_SOURCE_DIR}/../graphAPI
		${IE_MAIN_SOURCE_DIR}/src/inference_engine
		${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src		
        ${IE_MAIN_SOURCE_DIR}/include)

link_directories(${IE_MAIN_SOURCE_DIR}/${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

if (WIN32)
  target_link_libraries(${TARGET_NAME} format_reader inference_engine graphAPI pugixml)
else()
  target_link_libraries(${TARGET_NAME} format_reader inference_engine graphAPI pugixml dl pthread)
endif()

