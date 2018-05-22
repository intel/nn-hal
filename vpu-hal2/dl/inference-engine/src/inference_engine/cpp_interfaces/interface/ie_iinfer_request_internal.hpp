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

#include <memory>
#include <map>
#include <string>
#include <ie_common.h>
#include <ie_blob.h>

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in InferRequestBase forwarding mechanism
 */
class IInferRequestInternal {
public:
    typedef std::shared_ptr<IInferRequestInternal> Ptr;

    virtual ~IInferRequestInternal() = default;

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of IInferRequest while request is ongoing (running or waiting in queue)
     */
    virtual void Infer() = 0;

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  Note: not all plugins may provide meaningful data
     *  @param perfMap - a map of layer names to profiling information for that layer.
     */
    virtual void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const = 0;

    /**
     * @brief Set input/output data to infer
     * @note: Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    virtual void SetBlob(const char *name, const Blob::Ptr &data) = 0;

    /**
     * @brief Get input/output data to infer
     * @note: Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    virtual void GetBlob(const char *name, Blob::Ptr &data) = 0;
};

}  // namespace InferenceEngine
