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

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>

#include <ie_common.h>
#include <chrono>
#include <thread>
#include <functional>

#include <unistd.h>
#include <limits.h>
#include <stdlib.h>

#include "hddl_executor.h"
#include "vpu_logger.h"
#include "hddl_api.h"
#include "environment.h"

#include "hddl_infer_request.h"

#define HDDLCALL(call) {                                \
    int ret = (call);                                   \
    LOG_DEBUG("HDDLCALL: %s\n", #call);                 \
    if (HDDL_ERROR_NONE != ret) {                       \
        THROW_IE_EXCEPTION << #call " failed: " << ret; \
    }                                                   \
}

using namespace VPU::HDDLPlugin;
using namespace VPU::Common;
using namespace InferenceEngine;

Executor::Executor(const LoggerPtr &log) :
        _graphHandle(0), _deviceHandle(nullptr), _log(log), _num_stages(0) {
    _auxBuffer = {};
}

int Executor::openDevice() {
    _deviceHandle = hddlRegisterClient("HDDLPlugin");
    if (!_deviceHandle) {
        THROW_IE_EXCEPTION << "hddlRegisterClient failed";
    }
    HDDLCALL(hddlAllocatorInit(&_auxAllocatorHandle));
    return MYRIAD_X;
}

void Executor::allocateGraph(const std::vector<char> &graphFileContent, size_t numStages,
                             const char* networkName) {
    HDDLCALL(hddlCreateMvGraphFromMemory(_deviceHandle, networkName,
                                         graphFileContent.data(),
                                         graphFileContent.size(), &_graphHandle));

    // allocate aux data buffer
#define DEBUG_BUFFER_SIZE 120
#define THERMAL_BUFFER_SIZE 100
#define THERMAL_THROTTLING_LEVEL_SIZE sizeof(int)
#define COUNT_OF_ADDITIONAL_TIMINGS 3

    _num_stages = numStages;
    // FIXME: is there a way to get auxSize without such hardcoding?
    size_t auxSize = DEBUG_BUFFER_SIZE + THERMAL_BUFFER_SIZE + THERMAL_THROTTLING_LEVEL_SIZE
                     + (_num_stages + COUNT_OF_ADDITIONAL_TIMINGS) * sizeof(float);

    HDDLCALL(hddlAllocatorAlloc(_auxAllocatorHandle, auxSize, &_auxBuffer));
}

void Executor::InferSync(HddlBuffer *input, HddlBuffer *result, uint64_t *taskHandle) {
    HDDLCALL(hddlInferTaskSync(_deviceHandle, _graphHandle, "taskName", input, result, &_auxBuffer,
                               taskHandle));

    void *auxData = nullptr;
    HDDLCALL(hddlAllocatorGetBufferData(_auxAllocatorHandle, &_auxBuffer, &auxData));
    size_t auxSize = 0;
    HDDLCALL(hddlAllocatorGetBufferSize(_auxAllocatorHandle, &_auxBuffer, &auxSize));

    uint8_t *totalDeviceTimePtr = reinterpret_cast<uint8_t *>(auxData) + auxSize - sizeof(float);
    float totalDeviceTime = *reinterpret_cast<float *>(totalDeviceTimePtr);
    LOG_INFO("Executor::InferSync: taskHandle %lu: TotalDeviceTime = %f", *taskHandle, totalDeviceTime);
}

void Executor::InferAsync(HddlBuffer *input, HddlBuffer *result, uint64_t *taskHandle, void *userData) {
    auto callback =
            [](HddlTaskHandle taskHandle, int statusCode, void *userData) {
                (void)taskHandle; (void)statusCode;
                HDDLInferRequest *req = reinterpret_cast<HDDLInferRequest *>(userData);
                return req->HDDLCallback();
            };
    HDDLCALL(hddlInferTaskAsync(_deviceHandle, _graphHandle, "taskName", input, result, &_auxBuffer,
            callback, userData, taskHandle));
}

int Executor::Wait(uint64_t taskHandle, int64_t timeout) {
    int status = hddlWaitTask(_deviceHandle, taskHandle, timeout);

    uint8_t *totalDeviceTimePtr = reinterpret_cast<uint8_t *>(_auxBuffer.mapData) + _auxBuffer.size - sizeof(float);
    float totalDeviceTime = *reinterpret_cast<float *>(totalDeviceTimePtr);
    LOG_INFO("Executor::Wait: taskHandle %lu: TotalDeviceTime = %f", taskHandle, totalDeviceTime);

    return status;
}

void Executor::closeDevice() {
    if (_graphHandle) {
        HDDLCALL(hddlDestroyMvGraph(_deviceHandle, _graphHandle));
    }
    if (_deviceHandle)
        HDDLCALL(hddlUnregisterClient(_deviceHandle));

    void *mapData = nullptr;
    HDDLCALL(hddlAllocatorGetBufferData(_auxAllocatorHandle, &_auxBuffer, &mapData));
    if (mapData)
        HDDLCALL(hddlAllocatorFree(_auxAllocatorHandle, &_auxBuffer));

    if (_auxAllocatorHandle)
        HDDLCALL(hddlAllocatorDeinit(_auxAllocatorHandle));

    _graphHandle = 0;
    _deviceHandle = nullptr;
    _auxAllocatorHandle = 0;
}

Executor::~Executor() {
    closeDevice();
}

void Executor::printThrottlingStatus() {
    void *auxData = nullptr;
    HDDLCALL(hddlAllocatorGetBufferData(_auxAllocatorHandle, &_auxBuffer, &auxData));
    uint8_t *throttlingPtr = reinterpret_cast<uint8_t *>(auxData) + DEBUG_BUFFER_SIZE + THERMAL_BUFFER_SIZE;
    int throttling = *reinterpret_cast<int *>(throttlingPtr);
    if (throttling == 1) {
        LOG_INFO("\n** Device %p NCS temperature high - thermal throttling initiated **\n", _deviceHandle);
    } else if (throttling == 2) {
        LOG_WARNING("*********************** WARNING *************************\n"\
                    "  Device %p NCS temperature critical\n"                     \
                    "  Aggressive thermal throttling initiated\n"                \
                    "  Continued use may result in device damage\n"              \
                    "*********************************************************", _deviceHandle);
    }
}

std::shared_ptr<GraphInfo<float>> Executor::getPerfTimeInfo() {
    uint8_t *auxData = nullptr;
    HDDLCALL(hddlAllocatorGetBufferData(_auxAllocatorHandle, &_auxBuffer, reinterpret_cast<void **>(&auxData)));
    float *timeInfo = reinterpret_cast<float *>(auxData + DEBUG_BUFFER_SIZE + THERMAL_BUFFER_SIZE
                                                 + THERMAL_THROTTLING_LEVEL_SIZE);

    // TODO: why not to pass a vector<float> to this function?
    return std::make_shared<GraphInfo<float>>(timeInfo, _num_stages + COUNT_OF_ADDITIONAL_TIMINGS);
}
