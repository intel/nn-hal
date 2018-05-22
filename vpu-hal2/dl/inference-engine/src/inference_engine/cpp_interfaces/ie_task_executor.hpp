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

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include "ie_api.h"
#include "details/ie_exception.hpp"
#include "cpp_interfaces/ie_task_synchronizer.hpp"
#include "cpp_interfaces/ie_task.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_itask_executor.hpp"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(TaskExecutor) : public ITaskExecutor {
public:
    typedef std::shared_ptr<TaskExecutor> Ptr;

    TaskExecutor(std::string name = "Default");

    ~TaskExecutor();

    /**
     * @brief Add task for execution and notify working thread about new task to start.
     * @note can be called from multiple threads - tasks will be added to the queue and executed one-by-one in FIFO mode.
     * @param task - shared pointer to the task to start
     *  @return true if succeed to add task, otherwise - false
     */
    bool startTask(Task::Ptr task) override;

private:
    std::shared_ptr<std::thread> _thread;
    std::mutex _queueMutex;
    std::condition_variable _queueCondVar;
    std::queue<Task::Ptr> _taskQueue;
    bool _isStopped;
    std::string _name;
};

}  // namespace InferenceEngine
