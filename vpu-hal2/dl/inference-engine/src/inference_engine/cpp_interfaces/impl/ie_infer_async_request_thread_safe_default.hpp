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
#include <list>
#include <string>
#include <mutex>
#include <exception>
#include <cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp>
#include <cpp_interfaces/ie_task_with_stages.hpp>
#include <cpp_interfaces/ie_task_executor.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include "ie_infer_async_request_thread_safe_internal.hpp"

namespace InferenceEngine {

class AsyncInferRequestThreadSafeDefault : public AsyncInferRequestThreadSafeInternal {
public:
    typedef std::shared_ptr<AsyncInferRequestThreadSafeDefault> Ptr;

    explicit AsyncInferRequestThreadSafeDefault(IInferRequestInternal::Ptr request,
                                                const ITaskExecutor::Ptr &taskExecutor,
                                                const TaskSynchronizer::Ptr &taskSynchronizer,
                                                const ITaskExecutor::Ptr &callbackExecutor)
            : _syncRequest(request),
              _requestExecutor(taskExecutor),
              _callbackExecutor(callbackExecutor),
              _requestSynchronizer(taskSynchronizer),
              _callback(nullptr) {
        _syncTask = std::make_shared<Task>([this]() { _syncRequest->Infer(); });
        _currentTask = _syncTask;
    }

    virtual ~AsyncInferRequestThreadSafeDefault() {
        waitAllAsyncTasks();
    }

    void waitAllAsyncTasks() {
        try {
            while (!_listAsyncTasks.empty()) {
                _listAsyncTasks.remove_if([this](StagedTask::Ptr task) -> bool {
                    auto sts = task->getStatus();
                    return !task->isOnWait() && (Task::Status::TS_DONE == sts || Task::Status::TS_ERROR == sts ||
                                                 Task::Status::TS_INITIAL == sts);
                });
                auto findIter = std::find_if(_listAsyncTasks.begin(), _listAsyncTasks.end(),
                                             [this](StagedTask::Ptr task) { return !task->isOnWait(); });
                if (findIter != _listAsyncTasks.end()) {
                    try {
                        (*findIter)->wait(-1);
                    } catch (...) {}
                }
            }
        } catch (...) {}
    }

    virtual void initNextAsyncTask() {
        IE_PROFILING_AUTO_SCOPE(initNextAsyncTask)
        // Most probably was called from callback (or when callback was started) or it was a sync task before, so new task is required
        if (_currentTask->getStatus() == Task::Status::TS_POSTPONED || _currentTask == _syncTask) {
            auto findIter = std::find_if(_listAsyncTasks.begin(), _listAsyncTasks.end(),
                                         [this](StagedTask::Ptr task) -> bool {
                                             return (!task->isOnWait()) && (task != _currentTask) &&
                                                    (Task::Status::TS_DONE == task->getStatus() ||
                                                     Task::Status::TS_ERROR == task->getStatus());
                                         });
            if (findIter == _listAsyncTasks.end()) {
                _asyncTask = createAsyncRequestTask();
                _listAsyncTasks.push_back(_asyncTask);
            } else {
                _asyncTask = *findIter;
            }
        }
        _asyncTask->resetStages();
        _currentTask = _asyncTask;
    }

    virtual void startAsyncTask() {
        if (!_requestExecutor->startTask(_currentTask)) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
    }

    void StartAsync_ThreadUnsafe() override {
        initNextAsyncTask();
        startAsyncTask();
    }

    virtual StagedTask::Ptr createAsyncRequestTask() {
        return std::make_shared<StagedTask>([this]() {
            auto asyncTaskCopy = _asyncTask;
            try {
                switch (asyncTaskCopy->getStage()) {
                    case 2: {
                        _syncRequest->Infer();
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
        }, 2);
    }

    StatusCode Wait(int64_t millis_timeout) override {
        auto taskCopy = _currentTask;
        if (millis_timeout < IInferRequest::WaitMode::RESULT_READY) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str + "Timeout can't be less "
                               << IInferRequest::WaitMode::RESULT_READY
                               << " for InferRequest::Wait\n";
        }
        Task::Status status;
        if (millis_timeout == IInferRequest::WaitMode::STATUS_ONLY) {
            status = taskCopy->getStatus();
        } else {
            status = taskCopy->wait(millis_timeout);
            setIsRequestBusy(false);
        }

        taskCopy->checkException();
        return Task::TaskStatus2StatusCode(status);
    }

    void Infer_ThreadUnsafe() override {
        _currentTask = _syncTask;
        auto status = _currentTask->runWithSynchronizer(_requestSynchronizer);
        if (status == Task::Status::TS_BUSY)
            THROW_IE_EXCEPTION << "Internal error: AsyncInferRequestThreadSafeDefault failed to start sync task";
        _currentTask->checkException();
    }

    void GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const override {
        _syncRequest->GetPerformanceCounts(perfMap);
    }

    void SetBlob_ThreadUnsafe(const char *name, const Blob::Ptr &data) override {
        _syncRequest->SetBlob(name, data);
    }

    void GetBlob_ThreadUnsafe(const char *name, Blob::Ptr &data) override {
        _syncRequest->GetBlob(name, data);
    }

    void SetCompletionCallback_ThreadUnsafe(InferenceEngine::IInferRequest::CompletionCallback callback) override {
        _callback = callback;
    }

    void GetUserData_ThreadUnsafe(void **data) override {
        if (data == nullptr) THROW_IE_EXCEPTION << NOT_ALLOCATED_str;
        *data = _userData;
    }

    void SetUserData_ThreadUnsafe(void *data) override {
        _userData = data;
    }

    void SetPointerToPublicInterface(InferenceEngine::IInferRequest::Ptr ptr) {
        _publicInterface = ptr;
    }

protected:
    ITaskExecutor::Ptr _requestExecutor;
    ITaskExecutor::Ptr _callbackExecutor;
    TaskSynchronizer::Ptr _requestSynchronizer;
    IInferRequestInternal::Ptr _syncRequest;
    Task::Ptr _syncTask;
    StagedTask::Ptr _asyncTask;
    Task::Ptr _currentTask;
    std::list<StagedTask::Ptr> _listAsyncTasks;
    InferenceEngine::IInferRequest::CompletionCallback _callback;
    InferenceEngine::IInferRequest::WeakPtr _publicInterface;
    void *_userData;
};

}  // namespace InferenceEngine
