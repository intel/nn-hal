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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <map>
#include <cpp_interfaces/exception2status.hpp>

#include "hddl_api.h"
#include "hddl_plugin.h"
#include "environment.h"
#include <IExecutor.h>

namespace VPU {
namespace HDDLPlugin {

class Executor : public Common::IExecutor {
    void *_deviceHandle;
    uint64_t _graphHandle;
    HddlBuffer _auxBuffer;
    HddlAllocatorHandle _auxAllocatorHandle;
    size_t _num_stages;
    Common::LoggerPtr _log;

public:
    Executor(const Common::LoggerPtr &log);

    virtual ~Executor() noexcept(false);

    int openDevice() override;

    void closeDevice() override;

    void allocateGraph(const std::vector<char> &graphFileContent, size_t numStages, const char* networkName) override;

    void InferSync(HddlBuffer *input, HddlBuffer *result, uint64_t *taskHandle);

    void InferAsync(HddlBuffer *input, HddlBuffer *result, uint64_t *taskHandle, void *userData);

    int Wait(uint64_t taskHandle, int64_t timeout);

    void printThrottlingStatus() override;

    std::shared_ptr<Common::GraphInfo<float>> getPerfTimeInfo() override;
};

typedef std::shared_ptr<Executor> ExecutorPtr;

}  // namespace HDDLPlugin
}  // namespace VPU
