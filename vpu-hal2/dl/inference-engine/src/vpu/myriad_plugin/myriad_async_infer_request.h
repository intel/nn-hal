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

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "myriad_infer_request.h"

namespace VPU {
namespace MyriadPlugin {

class MyriadAsyncInferRequest : virtual public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    MyriadAsyncInferRequest(MyriadInferRequest::Ptr request,
                                const InferenceEngine::ITaskExecutor::Ptr &taskExecutorStart,
                                const InferenceEngine::ITaskExecutor::Ptr &taskExecutorGetResult,
                                const InferenceEngine::TaskSynchronizer::Ptr &taskSynchronizer,
                                const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor);

    InferenceEngine::StagedTask::Ptr createAsyncRequestTask() override;

    ~MyriadAsyncInferRequest();
private:
    MyriadInferRequest::Ptr _request;
    InferenceEngine::ITaskExecutor::Ptr _taskExecutorGetResult;
};

}  // namespace MyriadPlugin
}  // namespace VPU
