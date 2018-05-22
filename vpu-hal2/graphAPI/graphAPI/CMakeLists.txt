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


cmake_minimum_required (VERSION 2.8)
set (TARGET_NAME  graphAPI)

file(GLOB
        TOOL_SRC
        *.cpp
        *.c
        )

file(GLOB
        TOOL_INCLUDE
        *.hpp
        *.h
        )

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})
include_directories(
	${IE_MAIN_SOURCE_DIR}/include
	${IE_MAIN_SOURCE_DIR}/src/inference_engine
	${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src
)

add_library(${TARGET_NAME} STATIC ${TOOL_SRC} ${TOOL_INCLUDE})

