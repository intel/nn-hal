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
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"
#include "cpp_interfaces/base/ie_infer_async_request_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_async_request_internal.hpp"

namespace InferenceEngine {

class ExecutableNetworkThreadSafeAsyncOnly
        : public ExecutableNetworkInternal, public std::enable_shared_from_this<ExecutableNetworkThreadSafeAsyncOnly> {
public:
    typedef std::shared_ptr<ExecutableNetworkThreadSafeAsyncOnly> Ptr;

    virtual AsyncInferRequestInternal::Ptr
    CreateAsyncInferRequestImpl(InputsDataMap networkInputs, OutputsDataMap networkOutputs) = 0;

    void CreateInferRequest(IInferRequest::Ptr &asyncRequest) override {
        auto asyncRequestImpl = this->CreateAsyncInferRequestImpl(_networkInputs, _networkOutputs);
        asyncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        asyncRequest.reset(new InferRequestBase<AsyncInferRequestInternal>(asyncRequestImpl),
                           [](IInferRequest *p) { p->Release(); });
        asyncRequestImpl->SetPublicInterfacePtr(asyncRequest);
    }
};

}  // namespace InferenceEngine
