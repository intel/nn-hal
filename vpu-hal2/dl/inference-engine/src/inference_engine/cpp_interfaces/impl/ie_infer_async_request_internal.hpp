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
#include <cpp_interfaces/ie_task.hpp>
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in InferRequestBase forwarding mechanism
 */
class AsyncInferRequestInternal : public IAsyncInferRequestInternal, public InferRequestInternal {
public:
    typedef std::shared_ptr<AsyncInferRequestInternal> Ptr;

    explicit AsyncInferRequestInternal(InputsDataMap networkInputs, OutputsDataMap networkOutputs)
            : InferRequestInternal(networkInputs, networkOutputs), _callback(nullptr) {}

    void SetCompletionCallback(InferenceEngine::IInferRequest::CompletionCallback callback) {
        _callback = callback;
    }

    void GetUserData(void **data) {
        if (data == nullptr) THROW_IE_EXCEPTION << NOT_ALLOCATED_str;
        *data = _userData;
    }

    void SetUserData(void *data) {
        _userData = data;
    }

    /**
     * @brief Set weak pointer to the corresponding public interface: IInferRequest. This allow to pass it to
     * IInferRequest::CompletionCallback
     * @param ptr - weak pointer to InferRequestBase
     */
    void SetPublicInterfacePtr(IInferRequest::Ptr ptr) {
        _publicInterface = ptr;
    }

protected:
    IInferRequest::WeakPtr _publicInterface;
    InferenceEngine::IInferRequest::CompletionCallback _callback;
    void *_userData;
};

}  // namespace InferenceEngine
