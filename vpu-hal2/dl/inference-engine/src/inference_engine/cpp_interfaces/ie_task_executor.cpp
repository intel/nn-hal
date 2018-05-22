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

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <ie_profiling.hpp>
#include "details/ie_exception.hpp"
#include "ie_task.hpp"
#include "ie_task_executor.hpp"

namespace InferenceEngine {

TaskExecutor::TaskExecutor(std::string name) : _isStopped(false), _name(name) {
    _thread = std::make_shared<std::thread>([&] {
        while (!_isStopped) {
            bool isQueueEmpty;
            Task::Ptr currentTask;
            {  // waiting for the new task or for stop signal
                std::unique_lock<std::mutex> lock(_queueMutex);
                _queueCondVar.wait(lock, [&]() { return !_taskQueue.empty() || _isStopped; });
                isQueueEmpty = _taskQueue.empty();
                if (!isQueueEmpty) currentTask = _taskQueue.front();
            }
            if (_isStopped && isQueueEmpty)
                break;
            if (!isQueueEmpty) {
                currentTask->runNoThrowNoBusyCheck();
                std::unique_lock<std::mutex> lock(_queueMutex);
                _taskQueue.pop();
                isQueueEmpty = _taskQueue.empty();
                if (isQueueEmpty) {
                    // notify dtor, that all tasks were completed
                    _queueCondVar.notify_all();
                }
            }
        }
    });
}

TaskExecutor::~TaskExecutor() {
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (!_taskQueue.empty()) {
            _queueCondVar.wait(lock, [this]() { return _taskQueue.empty(); });
        }
        _isStopped = true;
        _queueCondVar.notify_all();
    }
    if (_thread && _thread->joinable()) {
        _thread->join();
        _thread.reset();
    }
}

bool TaskExecutor::startTask(Task::Ptr task) {
    if (!task->occupy()) return false;
    std::unique_lock<std::mutex> lock(_queueMutex);
    _taskQueue.push(task);
    _queueCondVar.notify_all();
    return true;
}

}  // namespace InferenceEngine
