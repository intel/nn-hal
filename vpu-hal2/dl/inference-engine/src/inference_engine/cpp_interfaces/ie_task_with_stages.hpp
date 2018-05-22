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
#include <mutex>
#include <memory>
#include <condition_variable>
#include <thread>
#include <queue>
#include "ie_api.h"
#include "details/ie_exception.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_task.hpp"
#include "ie_task_synchronizer.hpp"

namespace InferenceEngine {

/**
 * This class represents a task which can have several stages
 * and can be migrated from one task executor to another one
 * between stages. This is required to continue execution of the
 * task with special lock for device
 */
class INFERENCE_ENGINE_API_CLASS(StagedTask) : public Task {
public:
    typedef std::shared_ptr<StagedTask> Ptr;

    StagedTask(std::function<void()> function, size_t stages);

    StagedTask();

    Status runNoThrowNoBusyCheck() noexcept override;

    void resetStages();

    void stageDone();

    size_t getStage();

private:
    size_t _stages;
    size_t _stage;
};


}  // namespace InferenceEngine
