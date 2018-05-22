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
 * @brief A header file contains a wrapper class for handling plugin instantiation and releasing resources
 * @file ie_plugin_ptr.hpp
 */
#pragma once

#include "details/ie_so_pointer.hpp"
#include "ie_plugin.hpp"
#include "ie_ihetero_plugin.hpp"
#include <string>

namespace InferenceEngine {
namespace details {

/**
 * @class SOCreatorTrait
 * @brief This class defines the name of the fabric for creating an IInferencePlugin object in DLL
 */
template<>
class SOCreatorTrait<IInferencePlugin> {
public:
    /**
     * @brief A name of the fabric for creating IInferencePlugin object in DLL
     */
    static constexpr auto name = "CreatePluginEngine";
};

template<>
class SOCreatorTrait<IHeteroInferencePlugin> {
public:
    /**
     * @brief A name of the fabric for creating IInferencePlugin object in DLL
     */
    static constexpr auto name = "CreatePluginEngine";
};


}  // namespace details

/**
* @typedef InferenceEnginePluginPtr
* @brief A C++ helper to work with objects created by the plugin.
* Implements different interfaces.
*/
using InferenceEnginePluginPtr = InferenceEngine::details::SOPointer<IInferencePlugin>;
using HeteroPluginPtr = InferenceEngine::details::SOPointer<IHeteroInferencePlugin>;

}  // namespace InferenceEngine
