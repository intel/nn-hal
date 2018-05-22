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
 * @brief A header file for the TensorInfo structure
 * @file ie_tensor_info.hpp
 */

#pragma once

#include <string>
#include <memory>
#include <map>

namespace InferenceEngine {

struct TensorInfo {
    typedef std::shared_ptr<TensorInfo> Ptr;

    // memory layout BFYX, BXYF (enum)
    // size
    // precision
    std::map<std::string, std::string> extraInfo;
};

}  // namespace InferenceEngine
