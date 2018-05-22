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
 * @brief This is a header file for the Main Network reader class (wrapper) used to build networks from a given IR
 * @file ie_cnn_net_reader.h
 */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "ie_blob.h"
#include "ie_cnn_network.h"
#include "ie_common.h"
#include "ie_icnn_net_reader.h"
#include "details/ie_exception_conversion.hpp"

namespace InferenceEngine {
/**
 * @class CNNNetReader
 * @brief This is a wrapper class used to build and parse a network from the given IR.
 * All the methods here can throw exceptions.
 */
class CNNNetReader {
public:
    /**
     * @brief A smart pointer to this class
     */
    typedef std::shared_ptr<CNNNetReader> Ptr;

    /**
     * @brief A default constructor
     */
    CNNNetReader()
            : actual(shared_from_irelease(InferenceEngine::CreateCNNNetReader())) {
    }

    /**
    * @brief Parses the topology part of the IR (.xml).
    * Throws an exception in case of failure.
    * @param filepath The full path to the .xml file of the IR
    *
    */
    void ReadNetwork(const std::string &filepath) {
        CALL_STATUS_FNC(ReadNetwork, filepath.c_str());
    }

    /**
    * @brief Parses the topology part of the IR (.xml) given the xml as a buffer.
    * Throws an exception in case of failure.
    * @param model Pointer to a char array with the IR
    * @param size Size of the char array
    */
    void ReadNetwork(const void *model, size_t size) {
        CALL_STATUS_FNC(ReadNetwork, model, size);
    }

    /**
    * @brief Sets the weights buffer from the IR.
    * Weights Blob must always be of bytes - the casting to the precision is done per-layer to support mixed networks and for ease of use.
    * This method can be called more than once to reflect updates in the .bin.
    * Throws an exception in case of failure.
    * @param weights Blob of bytes that holds all the IR binary data
    */
    void SetWeights(const TBlob<uint8_t>::Ptr &weights) const {
        CALL_STATUS_FNC(SetWeights, weights);
    }

    /**
     * @brief Loads and sets the weights buffer directly from the IR .bin file.
     * Weights Blob must always be of bytes - the casting to the precision is done per-layer to support mixed networks and for ease of use.
     * This method can be called more than once to reflect updates in the .bin.
     * Throws an exception in case of failure.
     * @param filepath The full path to the .bin file
     */
    void ReadWeights(const std::string &filepath) const {
        CALL_STATUS_FNC(ReadWeights, filepath.c_str());
    }

    /**
    * @brief Gets a pointer to the built network
    * @return A reference to the CNNNetwork object to be loaded
     */
    CNNNetwork &getNetwork() {
        // network obj are to be updated upun this call
        if (network.get() == nullptr) {
            ICNNNetwork *icnn_network = actual->getNetwork(nullptr);
            if (icnn_network != nullptr) {
                network.reset(new CNNNetwork(icnn_network));
            } else {
                THROW_IE_EXCEPTION << "CNNNetwork::getNetwork: CNNNetwork was not initialized.";
            }
        }
        return *network.get();
    }

    /**
     * @brief Gets the flag that represents a status of model parsing
     * @return true if successful
     */
    bool isParseSuccess() const {
        CALL_FNC_NO_ARGS(isParseSuccess);
    }

    /**
     * @brief Gets the description of the current network
     * @return A string with the description
     */
    std::string getDescription() const {
        CALL_STATUS_FNC_NO_ARGS(getDescription);
        return resp.msg;
    }

    /**
     * @brief Gets a name of the current network
     * @return A string with the name
     */
    std::string getName() const {
        char name[64];
        CALL_STATUS_FNC(getName, name, sizeof(name) / sizeof(*name));
        return name;
    }

    /**
     * @brief Gets a version of the IR format of the current network
     * @return The version as an integer value
     */
    int getVersion() const {
        CALL_FNC_NO_ARGS(getVersion);
    }

private:
    std::shared_ptr<ICNNNetReader> actual;
    std::shared_ptr<CNNNetwork> network;
};
}  // namespace InferenceEngine
