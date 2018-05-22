//
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
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

#pragma once

#include <string>
#include <unordered_map>
#include "ie_api.h"
#include "cpp_interfaces/ie_itask_executor.hpp"

namespace InferenceEngine {

/**
 * @class ExecutorManagerImpl
 * @brief This class contains implementation of ExecutorManager global instance to provide task executor objects.
 * It helps to create isolated, independent unit tests for the its functionality. Direct usage of ExecutorManager class
 * (which is a singleton) makes it complicated.
 */
class ExecutorManagerImpl {
public:
    ITaskExecutor::Ptr getExecutor(std::string id);

    // for tests purposes
    size_t getExecutorsNumber();

    void clear();

private:
    std::unordered_map<std::string, ITaskExecutor::Ptr> executors;
};

/**
 * @class ExecutorManager
 * @brief This is global point for getting task executor objects by string id.
 * It's necessary in multiple asynchronous requests for having unique executors to avoid oversubscription.
 * E.g. There 2 task executors for CPU device: one - in FPGA, another - in MKLDNN. Parallel execution both of them leads to
 * not optimal CPU usage. More efficient to run the corresponding tasks one by one via single executor.
 */
class INFERENCE_ENGINE_API_CLASS(ExecutorManager) {
public:
    static ExecutorManager *getInstance() {
        if (!_instance) {
            _instance = new ExecutorManager();
        }

        return _instance;
    }

    ExecutorManager(ExecutorManager const &) = delete;

    void operator=(ExecutorManager const &)  = delete;

    /**
     * @brief Returns executor by unique identificator
     * @param id unique identificator of device (Usually string representation of TargetDevice)
     */
    ITaskExecutor::Ptr getExecutor(std::string id);

    // for tests purposes
    size_t getExecutorsNumber();

    // for tests purposes
    void clear();

private:
    ExecutorManager() {}

private:
    ExecutorManagerImpl _impl;
    static ExecutorManager *_instance;
};

}  // namespace InferenceEngine
