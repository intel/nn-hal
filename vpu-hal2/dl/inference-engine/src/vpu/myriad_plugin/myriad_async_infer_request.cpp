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

#include <memory>
#include "myriad_async_infer_request.h"

using namespace VPU::MyriadPlugin;
using namespace InferenceEngine;

MyriadAsyncInferRequest::MyriadAsyncInferRequest(MyriadInferRequest::Ptr request,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorStart,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorGetResult,
                                                 const InferenceEngine::TaskSynchronizer::Ptr &taskSynchronizer,
                                                 const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor)
        : InferenceEngine::AsyncInferRequestThreadSafeDefault(request,
                                                              taskExecutorStart,
                                                              taskSynchronizer,
                                                              callbackExecutor),
          _request(request), _taskExecutorGetResult(taskExecutorGetResult) {}


InferenceEngine::StagedTask::Ptr MyriadAsyncInferRequest::createAsyncRequestTask() {
    return std::make_shared<StagedTask>([this]() {
        auto asyncTaskCopy = _asyncTask;
        try {
            switch (asyncTaskCopy->getStage()) {
                case 3: {
                    _request->InferAsync();
                    asyncTaskCopy->stageDone();
                    _taskExecutorGetResult->startTask(asyncTaskCopy);
                }
                    break;
                case 2: {
                    _request->GetResult();
                    asyncTaskCopy->stageDone();
                    if (_callback) {
                        _callbackExecutor->startTask(asyncTaskCopy);
                    } else {
                        asyncTaskCopy->stageDone();
                    }
                }
                    break;
                case 1: {
                    auto requestPtr = _publicInterface.lock();
                    if (!requestPtr) {
                        THROW_IE_EXCEPTION << "Failed to run callback: can't get pointer to request";
                    }
                    setIsRequestBusy(false);
                    _callback(requestPtr, Task::TaskStatus2StatusCode(asyncTaskCopy->getStatus()));
                    asyncTaskCopy->stageDone();
                }
                    break;
                default:
                    break;
            }
        } catch (...) {
            setIsRequestBusy(false);
            std::rethrow_exception(std::current_exception());
        }
    }, 3);
}

MyriadAsyncInferRequest::~MyriadAsyncInferRequest() {
    waitAllAsyncTasks();
}
