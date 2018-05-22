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
 * @brief A header file for the PrimitiveInfo struct
 * @file ie_primitive_info_request.hpp
 */

#pragma once

#include "ie_tensor_info.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace InferenceEngine {

struct PrimitiveInfo {
    typedef std::shared_ptr<PrimitiveInfo> Ptr;

    std::string sId;          // some internal id, could be used as a name
    std::string sType;        // implementation type of this kernel
    int iPreAllocatedMemory;  // mainly the allocation of the output tensor

    std::vector<TensorInfo::Ptr> inputs;
    std::vector<TensorInfo::Ptr> outputs;

    std::map<std::string, std::string> extraInfo;  // any other important textual information user might find interesting about this kernel
};

}  // namespace InferenceEngine
