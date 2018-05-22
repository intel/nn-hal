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
 * @brief a header file for ExecutableNetwork wrapper over IExecutableNetwork
 * @file ie_iexecutable_network.hpp
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "ie_iexecutable_network.hpp"
#include "cpp/ie_infer_request.hpp"
#include "details/ie_exception_conversion.hpp"


namespace InferenceEngine {

/**
 * @class ExecutableNetwork
 * @brief wrapper over api IExecutableNetwork
 */
class ExecutableNetwork {
    IExecutableNetwork::Ptr actual;

public:
    ExecutableNetwork() = default;

    explicit ExecutableNetwork(IExecutableNetwork::Ptr actual) : actual(actual) {}

    /**
     * @brief Creates an asynchronous inference request object used to infer the network
     * @note: The returned request has allocated input and output blobs (that can be changed later)
     * @return InferRequest object
     */
    InferRequest CreateInferRequest() {
        IInferRequest::Ptr req;
        CALL_STATUS_FNC(CreateInferRequest, req);
        return InferRequest(req);
    }

    /**
     * @brief Creates an asynchronous inference request object used to infer the network
     * @note: The returned request has allocated input and output blobs (that can be changed later)
     * @return Shared pointer to the created request object
     */
    InferRequest::Ptr CreateInferRequestPtr() {
        IInferRequest::Ptr req;
        CALL_STATUS_FNC(CreateInferRequest, req);
        return std::make_shared<InferRequest>(req);
    }

    /**
    * @brief Exports the current executable network so it can be used later in the Import() main API
    * @param modelFileName Full path to the location of the exported file
    * @param resp Optional: pointer to an already allocated object to contain information in case of failure
    */
    void Export(const std::string &modelFileName) {
        CALL_STATUS_FNC(Export, modelFileName);
    }

    /**
    * @brief Gets the mapping of IR layer names to implemented kernels
    * @param deployedTopology Map of PrimitiveInfo objects that represent the deployed topology
    * @param resp Optional: pointer to an already allocated object to contain information in case of failure
    */
    void GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology) {
        CALL_STATUS_FNC(GetMappedTopology, deployedTopology);
    }

    /**
    * cast operator is used when this wrapper initialized by LoadNetwork
    * @return
    */
    operator IExecutableNetwork::Ptr &() {
        return actual;
    }

    typedef std::shared_ptr<ExecutableNetwork> Ptr;
};

}  // namespace InferenceEngine
