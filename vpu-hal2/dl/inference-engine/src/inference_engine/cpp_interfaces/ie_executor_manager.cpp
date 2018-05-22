//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#include <memory>
#include <string>
#include "cpp_interfaces/ie_executor_manager.hpp"
#include "cpp_interfaces/ie_task_executor.hpp"

namespace InferenceEngine {

ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(std::string id) {
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
        auto newExec = std::make_shared<TaskExecutor>(id);
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

// for tests purposes
size_t ExecutorManagerImpl::getExecutorsNumber() {
    return executors.size();
}

void ExecutorManagerImpl::clear() {
    executors.clear();
}

ExecutorManager *ExecutorManager::_instance = nullptr;

ITaskExecutor::Ptr ExecutorManager::getExecutor(std::string id) {
    return _impl.getExecutor(id);
}

size_t ExecutorManager::getExecutorsNumber() {
    return _impl.getExecutorsNumber();
}

void ExecutorManager::clear() {
    _impl.clear();
}

}  // namespace InferenceEngine
