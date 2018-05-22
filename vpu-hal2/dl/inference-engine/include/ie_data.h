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
 * @brief This header file defines the main Data representation node
 * @brief ie_data.h
 */
#pragma once

#include <map>
#include <memory>
#include <vector>
#include "ie_api.h"
#include "ie_common.h"
#include "details/ie_exception.hpp"
#include "ie_precision.hpp"
#include "ie_layouts.h"
#include <string>

namespace InferenceEngine {
/**
 * @class Data
 * @brief This class represents the main Data representation node, NN graphs are di-graphs: data nodes and layer nodes.
 */
class INFERENCE_ENGINE_API_CLASS(Data) {
public:
    /**
     * @deprecated Deprecated. Please use getPrecision()
     * @brief A precision type of this Data instance
     */
    Precision precision;
    /**
     * @deprecated Deprecated. Please use getFormat()
     * @brief A data layout of this Data instance
     */
    Layout layout;
    /**
     * @deprecated Deprecated. Please use getDims()
     * @brief A tensor dimension array (the order is opposite to the order in the IR: w,h,c,n) of this Data instance
     */
    SizeVector dims;
    /**
     * @deprecated Deprecated. Please use getCreatorLayer()
     * @brief A pointer to the layer that creates this data element, null for input data elements
     */
    CNNLayerWeakPtr creatorLayer;
    /**
     * @deprecated Deprecated. Please use getName()
     * @brief A unique name that identifies this data node
     */
    std::string name;
    /**
     * @deprecated Deprecated. Please use getInputTo()
     * @brief A map of layers that use this node as input.
     * It is useful for recursive NN graph traversal.
     */
    std::map<std::string, CNNLayerPtr> inputTo;
    /**
     * @deprecated Deprecated. Please use getUserObject()
     * @brief A user utility place holder
     */
    UserValue userObject;

    /**
     * @brief An empty constructor (dimensionless)
     * @param name Name of the data node
     * @param _precision Precision of the data
     */
    Data(const std::string &name, Precision _precision, Layout layout = NCHW);

    /**
     * @brief A full constructor (with dimensions)
     * @param name Name of the data node
     * @param a_dims Data tensor dimensions
     * @param _precision Precision of the data
     */
    Data(const std::string &name, const SizeVector &a_dims, Precision _precision, Layout layout = NCHW);
    /**
     * @brief A constructor with tensor descriptor
     * @param name Name of the data node
     * @param desc Tensor descriptor
     */
    Data(const std::string &name, const TensorDesc& desc);

    /**
     * @brief Checks if the current node is resolved
     * @return true if resolved, false otherwise.
     */
    bool isInitialized() const;

    /**
     * @brief Sets the data dimensions.
     * After that current node is marked as resolved.
     * @param a_dims Tensor dimensions to set
     */
    void setDims(const SizeVector &a_dims);

    /**
    * @brief Sets the batch value in the data dimensions.
    * Batch is defined as the last element in the dimensions vector.
    * @param batch_size Batch size to set
    */
    inline void setBatchSize(size_t batch_size);

    /**
    * @brief Sets the layout value for this Data instance
    * @param layout Layout value to set
    */
    void setLayout(Layout layout);
    Layout getLayout() const;

    const TensorDesc& getTensorDesc() const;
    /**
     * @brief Gets a precision type of this Data instance
     * @return Precision type
     */
    const Precision& getPrecision() const;
    void setPrecision(const Precision& precision);
    const SizeVector& getDims() const;

    CNNLayerWeakPtr& getCreatorLayer();

    const std::string& getName() const;
    std::map<std::string, CNNLayerPtr>& getInputTo();
    const UserValue& getUserObject() const;
private:
    TensorDesc tensorDesc;
};
}  // namespace InferenceEngine
