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
#include <mvnc.h>
#include <iomanip>
#include <environment.h>
#include <IExecutor.h>

namespace VPU {
namespace MyriadPlugin {

struct GraphDesc {
    graphHandle_t *_graphHandle = nullptr;

    ncTensorDescriptor_t *_inputDesc = nullptr;
    ncTensorDescriptor_t *_outputDesc = nullptr;

    fifoHandle_t *_inputFifoHandle = nullptr;
    fifoHandle_t *_outputFifoHandle = nullptr;
};

#define DEVICE_MAX_GRAPHS 2

struct DeviceDesc {
    int _executors = 0;
    int _platform = UNKNOWN_DEVICE;
    int _deviceIdx = -1;
    deviceHandle_t *_deviceHandle = nullptr;
};

typedef std::shared_ptr<DeviceDesc> DevicePtr;


class MyriadExecutor {
    Common::LoggerPtr _log;

public:
    MyriadExecutor(const Common::LogLevel& vpuLogLevel, const Common::LoggerPtr& log);
    ~MyriadExecutor();

    DevicePtr openDevice(std::vector<DevicePtr> &devicePool);

    static void closeDevices(std::vector<DevicePtr> &devicePool);

    void allocateGraph(DevicePtr &device, GraphDesc &graphDesc, const std::vector<char> &graphFileContent, size_t numStages, const char* networkName);

    void deallocateGraph(DevicePtr &device, GraphDesc &graphDesc);

    void queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                        void **result_data, size_t *result_bytes);

    void getResult(GraphDesc &graphDesc, void **result_data, size_t *result_bytes);

    const char *ncStatusToStr(graphHandle_t *graphHandle, ncStatus_t status);

    std::shared_ptr<Common::GraphInfo<float>> getPerfTimeInfo(graphHandle_t *graphHandle);

    void printThrottlingStatus();

    template<typename T>
    std::shared_ptr<Common::GraphInfo<T>> getGraphInfo(graphHandle_t *graphHandle, ncOptionClass_t opClass, int graphOption) {
        T *graphInfo;
        unsigned int graphInfoLen;
        if (NC_OK != ncGraphGetOption(graphHandle, opClass, graphOption,
                                      reinterpret_cast<void **>(&graphInfo),
                                      &graphInfoLen)) {
            graphInfo = nullptr;
            graphInfoLen = 0;
        }
        graphInfoLen /= sizeof(T);
        return std::make_shared<Common::GraphInfo<T>>(graphInfo, graphInfoLen);
    }
};

typedef std::shared_ptr<MyriadExecutor> MyriadExecutorPtr;

}  // namespace MyriadPlugin
}  // namespace VPU

