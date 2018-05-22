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
 * @brief A header file that defines a wrapper class for handling extension instantiation and releasing resources
 * @file ie_extension.hpp
 */
#pragma once

#include "details/ie_so_pointer.hpp"
#include "ie_iextension.h"
#include "mkldnn/mkldnn_extension_ptr.hpp"
#include <string>
#include <memory>

namespace InferenceEngine {
namespace details {

/**
 * @class SOCreatorTrait
 * @brief The SOCreatorTrait class defines the name of the fabric for creating IExtension object in DLL
 */
template<>
class SOCreatorTrait<IExtension> {
public:
    /**
     * @brief A name of the fabric for creating an IExtension object in DLL
     */
    static constexpr auto name = "CreateExtension";
};

}  // namespace details

/**
 * @class Extension
 * @brief This class is a C++ helper to work with objects created using extensions.
 * Implements different interfaces.
 */
class Extension : public IExtension {
public:
    /**
   * @brief Loads extension from a shared library
   * @param name Logical name of the extension library (soname without .dll/.so/lib prefix)
   */
    explicit Extension(const std::string &name)
            : actual(name) {}


    /**
     * @brief Gets the extension version information
     * @param versionInfo A pointer to version info, set by the plugin
     */
    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override {
        actual->GetVersion(versionInfo);
    }

    /**
     * @brief Sets a log callback that is used to track what is going on inside
     * @param listener Logging listener
     */
    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override {
        actual->SetLogCallback(listener);
    }

    /**
     * @brief Cleans the resources up
     */
    void Unload() noexcept override {
        actual->Unload();
    }

    /**
     * @brief Does nothing since destruction is done via the regular mechanism
     */
    void Release() noexcept override {}

    /**
     * @brief Gets the array with types of layers which are included in the extension
     * @param types Types array
     * @param size Size of the types array
     * @param resp Response descriptor
     * @return Status code
     */
    StatusCode getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept override {
        return actual->getPrimitiveTypes(types, size, resp);
    }
    /**
     * @brief Gets the factory with implementations for a given layer
     * @param factory Factory with implementations
     * @param cnnLayer A layer to get the factory for
     * @param resp Response descriptor
     * @return Status code
     */
    StatusCode getFactoryFor(ILayerImplFactory *&factory, const CNNLayer *cnnLayer,
                                     ResponseDesc *resp) noexcept override {
        return actual->getFactoryFor(factory, cnnLayer, resp);
    }

protected:
    /**
    * @brief An SOPointer instance to the loaded templated object
    */
    InferenceEngine::details::SOPointer<IExtension> actual;
};


/**
 * @brief Creates a special shared_pointer wrapper for the given type from a specific shared module
 * @param name Name of the shared library file
 * @return shared_pointer A wrapper for the given type from a specific shared module
 */
template<>
inline std::shared_ptr<IExtension> make_so_pointer(const std::string &name) {
    try {
        return std::make_shared<Extension>(name);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        return std::make_shared<MKLDNNPlugin::MKLDNNExtension>(name);
    }
}

}  // namespace InferenceEngine
