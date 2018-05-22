# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set (TARGET_NAME "inference_engine")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")

file (GLOB LIBRARY_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.cpp
      )

file (GLOB LIBRARY_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/base/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/impl/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/interface/*.hpp
      )

addVersionDefines(ie_version.cpp CI_BUILD_NUMBER)

set (PUBLIC_HEADERS_DIR "${IE_MAIN_SOURCE_DIR}/include")

file (GLOB PUBLIC_HEADERS
       ${PUBLIC_HEADERS_DIR}/*.h
       ${PUBLIC_HEADERS_DIR}/*.hpp
      )

if(WIN32)
  add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)
endif()


# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

# Create shared library file from sources

add_library(${TARGET_NAME} SHARED
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})


target_link_libraries(${TARGET_NAME} PRIVATE pugixml ade ${CMAKE_DL_LIBS} ${INTEL_ITT_LIBS})

# Properties->C/C++->General->Additional Include Directories
target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR}
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                  "${IE_MAIN_SOURCE_DIR}/src/dumper")

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ade/ade/include")
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ade/common/include")

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})

# Properties->C/C++->General->Additional Include Directories
target_include_directories(${TARGET_NAME}_s PUBLIC ${PUBLIC_HEADERS_DIR}
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                  "${IE_MAIN_SOURCE_DIR}/src/dumper")

target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")
target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ade/ade/include")
target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ade/common/include")

target_compile_definitions(${TARGET_NAME}_s PUBLIC -DUSE_STATIC_IE)

# export targets

export(TARGETS ${TARGET_NAME} NAMESPACE IE:: FILE "${CMAKE_BINARY_DIR}/targets.cmake")

include(CMakePackageConfigHelpers)

configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/cmake/config.cmake.in"
        "${CMAKE_BINARY_DIR}/inferenceengine-config.cmake"
        INSTALL_DESTINATION "${CMAKE_BINARY_DIR}"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO)

set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)
