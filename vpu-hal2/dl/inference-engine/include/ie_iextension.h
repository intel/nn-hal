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
 * @brief This is a header file for Inference Engine Extension Interface
 * @file ie_extension.h
 */
#pragma once

#include <ie_icnn_network.hpp>

#include "ie_api.h"
#include "ie_device.hpp"
#include "ie_layers.h"
#include "ie_error.hpp"
#include "ie_version.hpp"
#include <vector>

#include "details/ie_no_copy.hpp"

#include <memory>


#if defined(_WIN32) && defined(IMPLEMENT_INFERENCE_EXTENSION_API)
#define INFERENCE_EXTENSION_API(TYPE) extern "C"  __declspec(dllexport) TYPE
#else
#define INFERENCE_EXTENSION_API(TYPE) INFERENCE_ENGINE_API(TYPE)
#define INFERENCE_EXTENSION_CDECL INFERENCE_ENGINE_CDECL
#endif


namespace InferenceEngine {

/**
 * @struct DataConfig
 * @brief This structure describes data configuration
 */
struct DataConfig {
    /**
     * @brief Format of memory descriptor
     */
    TensorDesc desc;
    /**
     * @brief Index of in-place memory. If -1 memory cannot be in-place
     */
    int inPlace = -1;
    /**
     * @brief Flag for determination of the constant memory. If layer contains all constant memory we can calculate it on the load stage.
     */
    bool constant = false;
};

/**
 * @struct LayerConfig
 * @brief This structure describes Layer configuration
 */
struct LayerConfig {
    /**
     * @brief Supported dynamic batch. If false, dynamic batch is not supported
     */
    bool dynBatchSupport = false;
    /**
     * @brief Vector of input data configs
     */
    std::vector<DataConfig> inConfs;
    /**
     * @brief Vector of output data configs
     */
    std::vector<DataConfig> outConfs;
};

/**
 * @class ILayerImpl
 * @brief This class provides interface for extension implementations
 */
class ILayerImpl {
public:
    typedef std::shared_ptr<ILayerImpl> Ptr;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImpl() {}
    /**
     * @brief Gets all supported configurations for the current layer
     * @param conf Vector with supported configurations
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept = 0;
    /**
     * @brief Initializes the implementation
     * @param config Selected supported configuration
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept = 0;
};

/**
 * @class ILayerExecImpl
 * @brief This class provides interface for the implementation with the custom execution code
 */
class ILayerExecImpl: public ILayerImpl {
public:
    /**
     * @brief Execute method
     * @param inputs Vector of blobs with input memory
     * @param outputs Vector of blobs with output memory
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode execute(std::vector<Blob::Ptr>& inputs,
                               std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept = 0;
};

/**
 * @class ILayerImplFactory
 * @brief This class provides interface for extension factories
 */
class ILayerImplFactory {
public:
    typedef std::shared_ptr<ILayerImplFactory> Ptr;
    typedef std::function<ILayerImpl *()> ImplCreator;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImplFactory() {}
    /**
     * @brief Sets output shapes by input shapes.
     * @param inShapes Shapes of all inputs coming in this layer
     * @param outShapes Generated shapes coming from this layer given the input
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getShapes(const std::vector<TensorDesc>& inShapes, std::vector<TensorDesc>& outShapes,
                                 ResponseDesc *resp) noexcept = 0;
    /**
     * @brief Gets all possible implementations for the given cnn Layer
     * @param impls the vector with implementations which is ordered by priority
     * @param resp response descriptor
     * @return status code
     */
    virtual StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc *resp) noexcept = 0;
};

/**
 * @class IExtension
 * @brief This class is the main extension interface
 */
class IExtension : public InferenceEngine::details::IRelease {
public:
    /**
     * @brief Gets extension version information and stores in versionInfo
     * @param versionInfo Pointer to version info, will be set by plugin
     */
    virtual void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept = 0;

    /**
     * @brief Sets logging callback.
     * Logging is used to track what is going on inside.
     * @param listener Logging sink
     */
    virtual void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept = 0;

    /**
     * @brief Cleans resources up
     */
    virtual void Unload() noexcept = 0;
    /**
     * @brief Gets the array with types of layers which are included in the extension 
     * @param types Array to store the layer types 
     * @param size Size of the layer types array
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept = 0;
    /**
     * @brief Gets the factory with implementations for type
     * @param factory Factory with implementations
     * @param type Type of layer to get the factory for
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer,
                                     ResponseDesc *resp) noexcept = 0;
};

using IExtensionPtr = std::shared_ptr<IExtension>;

/**
 * @brief Creates the default instance of the extension
 * @param ext Extension interface
 * @param resp Response descriptor
 * @return Status code
 */
INFERENCE_EXTENSION_API(StatusCode) CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept;

}  // namespace InferenceEngine
