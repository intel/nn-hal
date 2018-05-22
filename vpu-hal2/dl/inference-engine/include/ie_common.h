// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief This is a header file with common inference engine definitions
 * @file ie_common.h
 */
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <ostream>
#include <algorithm>
#include <details/ie_exception.hpp>

namespace InferenceEngine {
/**
 * @typedef SizeVector
 * @brief Represents tensor size.
 * The order is opposite to the order in Caffe*: (w,h,n,b) where the most frequently changing element in memory is the first one
 */
typedef std::vector<size_t> SizeVector;

/**
 * @class CNNLayer
 * @brief This class represents the convolutional generic layer
 */
class CNNLayer;

/**
 * @brief A smart pointer to CNNLayer
 */
typedef std::shared_ptr<CNNLayer> CNNLayerPtr;
/**
 * @brief A smart weak pointer to CNNLayer
 */
typedef std::weak_ptr<CNNLayer> CNNLayerWeakPtr;

/**
 * @brief The main data representation node
 */
class Data;

/**
 * @brief Smart pointer to Data
 */
typedef std::shared_ptr<Data> DataPtr;
/**
 * @brief Smart weak pointer to Data
 */
typedef std::weak_ptr<Data> DataWeakPtr;

/**
 * @brief A union to hold user values to enable binding of data per graph node
 */
typedef union UserValue {
    int v_int;
    float v_float;
    void *v_ptr;
} UserValue;

/**
 * @enum Layout
 * @brief Layouts the inference engine supports
 */
enum Layout : uint8_t {
    ANY = 0,           // "any" layout

    // I/O data layouts
    NCHW = 1,
    NHWC = 2,

    // weight layouts
    OIHW = 64,

    // bias layouts
    C = 96,

    // Single image layout (for mean image)
    CHW = 128,

    //for depth conv 2d
    #ifdef AKS
    IHWO = 160,
    #endif

    // 2D
    HW = 192,
    NC = 193,
    CN = 194,

    BLOCKED = 200,
};

/**
 * @struct InferenceEngineProfileInfo
 * @brief Represents basic inference profiling information per layer.
 * If the layer is executed using tiling, the sum time per each tile is indicated as the total execution time.
 * Due to parallel execution, the total execution time for all layers might be greater than the total inference time.
 */
struct InferenceEngineProfileInfo {
    /**
     * @brief Defines the general status of the layer
     */
    typedef enum {
        NOT_RUN,
        OPTIMIZED_OUT,
        EXECUTED
    } LayerStatus;

    LayerStatus status;
    /**
     * @brief The absolute time in microseconds this layer ran (in total)
     */
    long long realTime_uSec;
    /**
     * @brief The net host cpu time this layer ran
     */
    long long cpu_uSec;

    /**
     * @brief A type of the execution unit
     */
    char exec_type[256] = {};

    /**
     * @brief A layer type
     */
    char layer_type[256] = {};

    /**
     * @brief An execution index of the unit
     */
    unsigned execution_index;
};


/**
 * @enum StatusCode
 * @brief This enum contains codes for all possible return values of the interface functions
 */
enum StatusCode {
    OK = 0,
    GENERAL_ERROR = -1,
    NOT_IMPLEMENTED = -2,
    NETWORK_NOT_LOADED = -3,
    PARAMETER_MISMATCH = -4,
    NOT_FOUND = -5,
    OUT_OF_BOUNDS = -6,
    /*
     * @brief exception not of std::exception derived type was thrown
     */
    UNEXPECTED = -7,
    REQUEST_BUSY = -8,
    RESULT_NOT_READY = -9,
    NOT_ALLOCATED = -10,
    INFER_NOT_STARTED = -11
};

/**
 * @struct ResponseDesc
 * @brief This class represents debug information for an error
 */
struct ResponseDesc {
    /**
     * @brief character buffer to hold the related information
     */
    char msg[256] = {};
};
}  // namespace InferenceEngine

#if defined(_WIN32)
    #define __PRETTY_FUNCTION__ __FUNCSIG__
#else
    #define __PRETTY_FUNCTION__ __PRETTY_FUNCTION__
#endif
