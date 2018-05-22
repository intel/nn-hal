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
 * @brief A header file for the main MKL-DNN Extension API
 * @file mkldnn_extension.hpp
 */
#pragma once

#include <ie_iextension.h>

#include "mkldnn_generic_primitive.hpp"

namespace InferenceEngine {
namespace MKLDNNPlugin {

/**
 * @deprecated use new extensibility API
 * @class IMKLDNNExtension
 * @brief The IMKLDNNExtension class provides the main extension interface
 */
class IMKLDNNExtension : public IExtension {
public:
    /**
     * @brief Creates a generic layer and returns a pointer to an instance
     * @param primitive Pointer to newly created layer
     * @param layer Layer parameters (source for name, type, precision, attr, weights...)
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
     */
    virtual InferenceEngine::StatusCode CreateGenericPrimitive(IMKLDNNGenericPrimitive*& primitive,
                                          const InferenceEngine::CNNLayerPtr& layer,
                                          InferenceEngine::ResponseDesc *resp) const noexcept = 0;
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        return NOT_IMPLEMENTED;
    };
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer, ResponseDesc *resp) noexcept override {
        return NOT_IMPLEMENTED;
    }
};

/**
 * @deprecated use new extensibility API
 * @brief Creates the default instance of the extension
 * @return The MKL-DNN Extension interface
 */
INFERENCE_EXTENSION_API(StatusCode) CreateMKLDNNExtension(IMKLDNNExtension*& ext, ResponseDesc* resp) noexcept;

}  // namespace MKLDNNPlugin
}  // namespace InferenceEngine
