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
 * @brief This is a header file for the ICNNNetwork class
 * @file ie_icnn_network.hpp
 */
#pragma once

#include "ie_common.h"
#include "ie_layers.h"
#include "ie_data.h"
#include "ie_device.hpp"
#include "ie_blob.h"
#include "details/ie_irelease.hpp"
#include "ie_preprocess.hpp"
#include "ie_input_info.hpp"
#include <memory>
#include <map>
#include <string>

namespace InferenceEngine {

/**
 * @brief A map of names of outputs and a smart pointer to their values
 */
typedef std::map<std::string, DataPtr> OutputsDataMap;

/**
 * @class ICNNNetwork
 * @brief This is the main interface to describe the NN topology
 */
class ICNNNetwork : public details::IRelease {
public:
    /**
     * @brief Returns the main network operating precision.
     * This may be MIXED if not homogeneous.
     * @return A precision type
     */
    virtual Precision getPrecision() noexcept = 0;

    /**
     * @brief Gets the network output Data node information. The received info is stored in the given Data node.
     * For single and multiple outputs networks.
     * @param out Reference to the Data node smart pointer
     */
    virtual void getOutputsInfo(OutputsDataMap &out) const noexcept  = 0;

    /**
     * @brief Gets the network input Data node information. The received info is stored in the given Data node.
     * For single and multiple inputs networks.
     * This method must be called to find out input names for using them later during filling of a map
     * of blobs passed later to InferenceEngine::IInferencePlugin::Infer()
     * @param inputs Reference to map of pairs: (string, InputInfo smart pointer).
     */
    virtual void getInputsInfo(InputsDataMap &inputs) const noexcept  = 0;


    /**
     * @brief Returns information on certain input pointed by inputName
     * @param inputName Name of input layer to get info on
     * @return A smart pointer to the input information
     */
    virtual InputInfo::Ptr getInput(const std::string &inputName) noexcept = 0;

    /**
     * @brief Gets the network name. The name is stored in the given pName string.
     * @param pName - will receive actual network name, specified in IR file,
     *     pName should point to valid memory address before invoking this function
     * @param len - size in bytes of pName buffer, actual name is trimmed by this size
     */
    virtual void getName(char *pName, size_t len)  noexcept = 0;

    /**
    * @brief Returns the number of layers in the network as an integer value
    * @return The number of layers as an integer value
    */
    virtual size_t layerCount()  noexcept = 0;

    /**
    * @brief Returns a smart pointer to a Data node given its name. If the Data node is missing, returns an empty data pointer.
    * @param dname Name of the Data node
    * @return Data node smart pointer
    */
    virtual DataPtr &getData(const char *dname) noexcept = 0;

    /**
    * @brief Adds a layer to the network. A user is responsible to connect it to other data elements.
    * @param layer Const reference to a layer smart pointer
    */
    virtual void addLayer(const CNNLayerPtr &layer) noexcept = 0;

    /**
     * @brief Adds output to the layer
     * @param layerName Name of the layer
     * @param outputIndex Index of the output
     * @param resp Response message
     * @return Status code of the operation
     */
    virtual StatusCode
    addOutput(const std::string &layerName, size_t outputIndex = 0, ResponseDesc *resp = nullptr) noexcept = 0;

    /**
     * @brief Gets network layer with the given name
     * @param layerName Given name of the layer
     * @param out Pointer to the found CNNLayer object with the given name
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode getLayerByName(const char *layerName, CNNLayerPtr &out, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Sets a desirable device to perform all work on.
     * Some plug-ins might not support some target devices and may abort execution with an appropriate error message.
     * @param device Device to set as a target
     */
    virtual void setTargetDevice(TargetDevice device) noexcept = 0;

    /**
     * @brief Gets the target device.
     * If setTargetDevice() was not called before, returns eDefault
     * @return A TargetDevice instance
     */
    virtual TargetDevice getTargetDevice() noexcept = 0;

    /**
    * @brief Changes the inference batch size
    * @param size Size of batch to set
    * @return Status code of the operation
    */
    virtual StatusCode setBatchSize(const size_t size) noexcept = 0;

    /**
    * @brief Gets the inference batch size
    * @return The size of batch as a size_t value
    */
    virtual size_t getBatchSize() const noexcept = 0;
};
}  // namespace InferenceEngine
