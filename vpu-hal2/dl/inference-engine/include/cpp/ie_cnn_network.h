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
 * @brief A header file that defines the public API of the CNNNetwork object
 * @file ie_cnn_network.h
 */
#pragma once

#include <details/ie_exception_conversion.hpp>
#include <details/ie_cnn_network_iterator.hpp>
#include <ie_icnn_network.hpp>
#include "ie_common.h"
#include "ie_data.h"
#include "ie_blob.h"
#include <vector>
#include <string>
#include <map>
#include <utility>

namespace InferenceEngine {

/**
 * @class CNNNetwork
 * @brief This class contains all the information about the Neural Network and the related binary information
 */
class CNNNetwork {
public:
    /**
     * @brief A default constructor
     * @param actual Pointer to the network object
     */
    explicit CNNNetwork(ICNNNetwork *actual)
            : actual(actual) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
        }
    }

    /**
     * @brief Gets the main network operating precision
     * @return The precision instance
     */
    virtual Precision getPrecision() {
        return actual->getPrecision();
    }

    /**
    * @brief Gets the container which stores all network outputs
    * The container is defined as associations: (output name, smart pointer to the data node)
    * @return The associative container
    */
    virtual OutputsDataMap getOutputsInfo() {
        OutputsDataMap outputs;
        actual->getOutputsInfo(outputs);
        return std::move(outputs);
    }

    /**
    * @brief Gets the container that stores all network inputs
    * The container is defined as associations: (output name, smart pointer to the data node)
    * @return associative container
    */
    virtual InputsDataMap getInputsInfo() {
        InputsDataMap inputs;
        actual->getInputsInfo(inputs);
        return std::move(inputs);
    }

    /**
     * @brief Gets a number of layers in the network
     * @return A number of layers
     */
    size_t layerCount() const {
        return actual->layerCount();
    }

    /**
    * @brief Changes the inference batch size
    * @param size New size of batch
    */
    virtual void setBatchSize(const size_t size) {
        actual->setBatchSize(size);
    }

    /**
    * @brief Gets the inference batch size
    * @return The size of batch
    */
    virtual size_t getBatchSize() const {
        return actual->getBatchSize();
    }

    /**
     * @brief An overloaded operator & to get current network
     * @return An instance of the current network
     */
    operator ICNNNetwork &() const {
        return *actual;
    }

    /**
     * @brief Sets tha target device
     * @param device Device instance to set
     */
    void setTargetDevice(TargetDevice device) {
        actual->setTargetDevice(device);
    }

    /**
     * @brief Adds output to the layer
     * @param layerName Name of the layer to modify
     * @param outputIndex Index of the output
     */
    void addOutput(const std::string &layerName, size_t outputIndex = 0) {
        CALL_STATUS_FNC(addOutput, layerName, outputIndex);
    }

    /**
     * @brief Gets network layer corresponding to name given
     * @param layerName - given name of the layer
     * @param out - pointer to found CNNLayer with name equal to given one
     * @param resp - pointer to object, which would hold description of error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    CNNLayerPtr getLayerByName(const char *layerName) {
        CNNLayerPtr layer;
        CALL_STATUS_FNC(getLayerByName, layerName, layer);
        return layer;
    }
    /**
     * @brief iterator over cnn layers, order of layers is implementation specific, and can be changed in future
     */
    details::CNNNetworkIterator begin() const {
        return details::CNNNetworkIterator(actual);
    }

    details::CNNNetworkIterator end() const {
        return details::CNNNetworkIterator();
    }
    /**
     * @brief number of layers in network object
     * @return
     */
    size_t size() const {
        return std::distance(std::begin(*this), std::end(*this));
    }

protected:
    /**
     * @brief A pointer to the current network
     */
    ICNNNetwork *actual = nullptr;
    /**
     * @brief A pointer to output data
     */
    DataPtr output;
};

}  // namespace InferenceEngine
