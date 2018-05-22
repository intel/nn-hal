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
 * @brief a header file for IExecutableNetwork interface
 * @file ie_iexecutable_network.hpp
 */
#pragma once

#include "ie_common.h"
#include "ie_primitive_info.hpp"
#include "ie_iinfer_request.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace InferenceEngine {

/**
 * @class IExecutableNetwork
 * @brief This is an interface of an executable network
 */
class IExecutableNetwork : public details::IRelease {
public:
    typedef std::shared_ptr<IExecutableNetwork> Ptr;

    /**
    * @brief Creates an asynchronous inference request object used to infer the network.
    * The created request has allocated input and output blobs (that can be changed later).
    * @param req Shared pointer to the created request object
    * @param resp Optional: pointer to an already allocated object to contain information in case of failure
    * @return Status code of the operation: OK (0) for success
    */
    virtual StatusCode CreateInferRequest(IInferRequest::Ptr& req, ResponseDesc *resp) noexcept = 0;

    /**
    * @brief Exports the current executable network so it can be used later in the Import() main API
    * @param modelFileName Full path to the location of the exported file
    * @param resp Optional: pointer to an already allocated object to contain information in case of failure
    * @return Status code of the operation: OK (0) for success
    */
    virtual StatusCode Export(const std::string& modelFileName, ResponseDesc *resp) noexcept = 0;

    /**
    * @brief Gets the mapping of IR layer names to implemented kernels
    * @param deployedTopology Map of PrimitiveInfo objects that represent the deployed topology
    * @param resp Optional: pointer to an already allocated object to contain information in case of failure
    * @return Status code of the operation: OK (0) for success
    */
    virtual StatusCode GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology, ResponseDesc *resp) noexcept = 0;
};

}  // namespace InferenceEngine
