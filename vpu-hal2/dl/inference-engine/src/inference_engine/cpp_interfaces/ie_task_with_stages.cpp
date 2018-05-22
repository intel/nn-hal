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

#include <vector>
#include <memory>
#include <thread>
#include "details/ie_exception.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_task.hpp"
#include "cpp_interfaces/ie_task_with_stages.hpp"

namespace InferenceEngine {

StagedTask::StagedTask() : Task(), _stages(0) {}

StagedTask::StagedTask(std::function<void()> function, size_t stages) : Task(function), _stages(stages) {
    if (!function) THROW_IE_EXCEPTION << "Failed to create StagedTask object with null function";
    resetStages();
}

Task::Status StagedTask::runNoThrowNoBusyCheck() noexcept {
    try {
        _exceptionPtr = nullptr;
        if (_stage) {
            setStatus(TS_POSTPONED);
        }
        _function();
        if (!_stage) {
            setStatus(TS_DONE);
        }
    } catch (...) {
        _exceptionPtr = std::current_exception();
        setStatus(TS_ERROR);
    }

    if (_status != TS_POSTPONED) {
        _isTaskDoneCondVar.notify_all();
    }
    return getStatus();
}

void StagedTask::resetStages() {
    _stage = _stages;
}

void StagedTask::stageDone() {
    if (_stage <= 0) THROW_IE_EXCEPTION << "Failed to make stage done, because it's been already done";
    _stage--;
}

size_t StagedTask::getStage() {
    return _stage;
}

}  // namespace InferenceEngine
