// Copyright (c) 2017 Intel Corporation
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
 * @brief Convinience wrapper class for handling plugin instanciation and releasing resources.
 * @file ie_dump_plugin_ptr.hpp
 */
#pragma once

#include "details/ie_so_pointer.hpp"
#include "ie_dump_plugin.hpp"
#include <string>

namespace InferenceEngine {
namespace details {

template<>
class SOCreatorTrait<IDumpPlugin> {
public:
    static constexpr auto name = "CreateDumpPlugin";
};

}  // namespace details


}  // namespace InferenceEngine


/**
* @typedef DumpPluginPtr
* @brief c++ helper to work with plugin's created objects, implements different interface
*/
typedef InferenceEngine::details::SOPointer<IDumpPlugin> DumpPluginPtr;


