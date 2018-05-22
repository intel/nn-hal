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
 * @brief A header file that provides interface for network reader that is used to build networks from a given IR
 * @file ie_icnn_net_reader.h
 */
#pragma once

#include <map>
#include <string>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "details/ie_no_copy.hpp"
#include "ie_api.h"

namespace InferenceEngine {
/**
 * @class ICNNNetReader
 * @brief This class is the main interface to build and parse a network from a given IR
 * All methods here do not throw exceptions and return a ResponseDesc object.
 * Alternatively, to use methods that throw exceptions, refer to the CNNNetReader wrapper class.
 */
class ICNNNetReader : public details::IRelease {
public:
    /**
     * @brief Parses the topology part of the IR (.xml)
     * @param filepath The full path to the .xml file of the IR
     * @param resp Response message
     * @return Result code
     */
    virtual StatusCode ReadNetwork(const char *filepath, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Parses the topology part of the IR (.xml) given the xml as a buffer
     * @param model Pointer to a char array with the IR
     * @param resp Response message
     * @param size Size of the char array
     * @return Result code
     */
    virtual StatusCode ReadNetwork(const void *model, size_t size, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Sets the weights buffer (.bin part) from the IR.
     * Weights Blob must always be of bytes - the casting to precision is done per-layer to support mixed
     * networks and to ease of use.
     * This method can be called more than once to reflect update in the .bin.
     * @param weights Blob of bytes that holds all the IR binary data
     * @param resp Response message
     * @return Result code
    */
    virtual StatusCode SetWeights(const TBlob<uint8_t>::Ptr &weights, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Loads and sets the weights buffer directly from the IR .bin file.
     * This method can be called more than once to reflect updates in the .bin.
     * @param filepath Full path to the .bin file
     * @param resp Response message
     * @return Result code
     */
    virtual StatusCode ReadWeights(const char *filepath, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Returns a pointer to the built network
     * @param resp Response message
     */
    virtual ICNNNetwork *getNetwork(ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Retrieves the last building status
     * @param resp Response message
     */
    virtual bool isParseSuccess(ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Retrieves the last building failure message if failed
     * @param resp Response message
     * @return StatusCode that indicates the network status
     */
    virtual StatusCode getDescription(ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Returns network status by its name
     * @param name Name of the network
     * @param len Length of the name
     * @param resp Response message
     * @return StatusCode that indicates the network status
     */
    virtual StatusCode getName(char *name, size_t len, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Returns a version of IR
     * @param resp Response message
     * @return IR version number: 1 or 2
     */
    virtual int getVersion(ResponseDesc *resp) noexcept = 0;
};

/**
 * @brief Creates a CNNNetReader instance
 * @return An object that implements the ICNNNetReader interface
 */
INFERENCE_ENGINE_API(ICNNNetReader*)CreateCNNNetReader() noexcept;
}  // namespace InferenceEngine
