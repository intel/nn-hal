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

#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <ie_iinfer_request.hpp>
#include <ie_primitive_info.hpp>

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in ExecutableNetworkBase forwarding mechanism.
 */
class IExecutableNetworkInternal {
public:
    typedef std::shared_ptr<IExecutableNetworkInternal> Ptr;

    virtual ~IExecutableNetworkInternal() = default;

    /**
     * @brief Create an inference request object used to infer the network
     *  Note: the returned request will have allocated input and output blobs (that can be changed later)
     * @param req - shared_ptr for the created request
     */
    virtual void CreateInferRequest(IInferRequest::Ptr &req) = 0;

    /**
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param modelFileName - path to the location of the exported file
     */
    virtual void Export(const std::string &modelFileName) = 0;

    /**
     * @brief Get the mapping of IR layer names to actual implemented kernels
     * @param deployedTopology - map of PrimitiveInfo objects representing the deployed topology
     */
    virtual void GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology) = 0;
};

}  // namespace InferenceEngine
