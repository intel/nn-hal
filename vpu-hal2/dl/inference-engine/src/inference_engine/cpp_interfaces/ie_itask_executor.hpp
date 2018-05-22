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

#pragma once

#include <memory>
#include "ie_api.h"
#include "ie_task.hpp"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(ITaskExecutor) {
public:
    typedef std::shared_ptr<ITaskExecutor> Ptr;

    /**
     * @brief Add task for execution and notify working thread about new task to start.
     * @note can be called from multiple threads - tasks will be added to the queue and executed one-by-one in FIFO mode.
     * @param task - shared pointer to the task to start
     *  @return true if succeed to add task, otherwise - false
     */
    virtual bool startTask(Task::Ptr task) = 0;
};

}  // namespace InferenceEngine
