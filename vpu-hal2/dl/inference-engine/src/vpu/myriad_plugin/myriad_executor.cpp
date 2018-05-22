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
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <sys/stat.h>
#include <dirent.h>

#include <mvnc.h>
#include <ie_common.h>
#include <chrono>
#include <thread>
#include <vpu_logger.h>

#include "myriad_executor.h"
#include <cpp_interfaces/exception2status.hpp>

#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

using namespace VPU::Common;
using namespace VPU::MyriadPlugin;
using namespace InferenceEngine;
using namespace std;

static std::mutex device_mutex;

MyriadExecutor::MyriadExecutor(const LogLevel& vpuLogLevel, const LoggerPtr& log) :
        _log(log) {
    int ncLogLevel = 3;
    switch (vpuLogLevel) {
    case eLOGWARNING:
        ncLogLevel = 2;
        break;
    case eLOGINFO:
        ncLogLevel = 1;
        break;
    case eLOGDEBUG:
        ncLogLevel = 0;
        break;
    }

    auto status = ncGlobalSetOption(NC_RW_LOG_LEVEL, &ncLogLevel, sizeof(ncLogLevel));
    if (status != NC_OK) {
        LOG_WARNING("WARNING: failed to set log level: %d with error: %s\n",
                    ncLogLevel,
                    ncStatusToStr(nullptr, status));
    }
}

DevicePtr MyriadExecutor::openDevice(std::vector<DevicePtr> &devicePool) {
    std::lock_guard<std::mutex> lock(device_mutex);
    ncStatus_t statusInit = NC_ERROR;
    ncStatus_t statusOpen = NC_ERROR;

    // check already booted but empty devices
    int deviceIdx = -1;
    while (++deviceIdx < devicePool.size()) {
        if (devicePool[deviceIdx]->_executors == 0) {
            devicePool[deviceIdx]->_executors = 1;
            return devicePool[deviceIdx];
        }
    }

    // try to boot next device if any
    if (statusInit != NC_OK) {
        deviceIdx = devicePool.size();
        DeviceDesc device;
        statusInit = ncDeviceInit(deviceIdx, &device._deviceHandle);
        if (statusInit == NC_OK) {
            statusOpen = ncDeviceOpen(device._deviceHandle);
            if (statusOpen == NC_OK) {
                unsigned int dataLength = 0;
/*                if (NC_OK != ncDeviceGetOption(device._deviceHandle, NC_OPTION_CLASS0,
                        NC_RO_DEVICE_PLATFORM, reinterpret_cast<void*>(&device._platform), &dataLength)
                    || dataLength != sizeof(device._platform)) {
                    LOG_WARNING("WARNING: Failed to get device platform");
                }
*/
#ifdef AKS
                device._platform = 2450; //fixed for Myriad 2450
#endif
                device._executors = 1;
                device._deviceIdx = deviceIdx;
                devicePool.push_back(std::make_shared<DeviceDesc>(device));
            }
        }
    }

    // attach one more executor to already booted device
    if (statusInit != NC_OK) {
        deviceIdx = -1;
        while (++deviceIdx < devicePool.size()) {
            if (devicePool[deviceIdx]->_executors < DEVICE_MAX_GRAPHS) {
                devicePool[deviceIdx]->_executors += 1;
                return devicePool[deviceIdx];
            }
        }
    }

    if (statusInit != NC_OK) {
        THROW_IE_EXCEPTION << "Can not init USB device: " << ncStatusToStr(nullptr, statusInit);
    }
    if (statusOpen != NC_OK) {
        THROW_IE_EXCEPTION << "Can not open USB device: " << ncStatusToStr(nullptr, statusOpen);
    }
    if (devicePool[deviceIdx]->_platform == UNKNOWN_DEVICE) {
        THROW_IE_EXCEPTION << "Unknown device";
    }

    return devicePool[deviceIdx];
}

void MyriadExecutor::closeDevices(std::vector<DevicePtr> &devicePool) {
    std::lock_guard<std::mutex> lock(device_mutex);
    for (auto &device : devicePool) {
        if (device->_deviceHandle != nullptr) {
            auto res = ncDeviceClose(device->_deviceHandle);
            if (res != NC_OK) {
            #ifdef NNLOG
                ALOGI("ncDeviceClose failed\n");
            #endif
            // TODO:
            //    LOG_DEBUG("Close Device result %s.", ncStatusToStr(nullptr, res));
          }
            device->_deviceHandle = nullptr;
        }
    }

    #ifdef NNLOG
    ALOGI("MyriadExecutor::closeDevices done");
    #endif
}

void MyriadExecutor::allocateGraph(DevicePtr &device, GraphDesc &graphDesc,
        const std::vector<char> &graphFileContent, size_t numStages, const char* networkName) {

    LOG_INFO("MyriadExecutor::allocateGraph");
    if (device->_deviceHandle == nullptr) {
        LOG_INFO("MyriadExecutor::allocateGraph _deviceHandle is Null");
        THROW_IE_EXCEPTION << "Failed to allocate graph: MYRIAD device is not opened.";
    }


    #ifdef NNLOG
    ALOGI("MyriadExecutor::allocateGraph");
    #endif
    ncStatus_t status;

    status = ncGraphInit(networkName, &graphDesc._graphHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init graph: " << ncStatusToStr(nullptr, status);
    }
    int executors = device->_platform == MYRIAD_X ? 2 : 1;

    status = ncGraphSetOption(graphDesc._graphHandle, NC_OPTION_CLASS1, NC_RW_GRAPH_EXECUTORS_NUM, &executors, sizeof(executors));
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to set graph executors: " << ncStatusToStr(nullptr, status);
    }

    status = ncGraphAllocate(device->_deviceHandle, graphDesc._graphHandle, graphFileContent.data(), graphFileContent.size());
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to allocate graph: " << ncStatusToStr(nullptr, status);
    }

    LOG_INFO("MyriadExecutor::allocateGraph ncGraphAllocate done");
    #ifdef NNLOG
    ALOGI("MyriadExecutor::allocateGraph ncGraphAllocate done");
    #endif
    unsigned int dataLength = 0;

    int numInputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_OPTION_CLASS0, NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of inputs: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numInputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of inputs: " << numInputs;
    }

    int numOutputs = 0;
    status = ncGraphGetOption(graphDesc._graphHandle, NC_OPTION_CLASS0, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get number of outputs: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
    if (numOutputs != 1) {
        THROW_IE_EXCEPTION << "Unsupported number of outputs: " << numOutputs;
    }

    status = ncGraphGetOption(graphDesc._graphHandle, NC_OPTION_CLASS0, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &graphDesc._inputDesc, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get input description: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncGraphGetOption(graphDesc._graphHandle, NC_OPTION_CLASS0, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &graphDesc._outputDesc, &dataLength);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to get output description: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    int fifo_elements = 4;

    status = ncFifoInit(NC_FIFO_HOST_WO, &graphDesc._inputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init input FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoCreate(graphDesc._inputFifoHandle, device->_deviceHandle, graphDesc._inputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create input FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoInit(NC_FIFO_HOST_RO, &graphDesc._outputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to init output FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncFifoCreate(graphDesc._outputFifoHandle, device->_deviceHandle, graphDesc._outputDesc, fifo_elements);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to create output FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }
}

void MyriadExecutor::queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                    void **result_data, size_t *result_bytes) {
#ifndef NDEBUG
    if (auto dumpFileName = std::getenv("IE_VPU_DUMP_INPUT_FILE_NAME")) {
        std::ofstream file(dumpFileName, std::ios_base::binary | std::ios_base::out);
        if (!file.is_open()) {
            THROW_IE_EXCEPTION << "[VPU] Cannot open file " << dumpFileName << " for writing";
        }
        file.write(static_cast<const char*>(input_data), input_bytes);
    }
#endif

    LOG_INFO("MyriadExecutor::queueInference");
    #ifdef NNLOG
    ALOGI("MyriadExecutor::queueInference");
    #endif
    if (graphDesc._inputDesc->totalSize != input_bytes) {
        THROW_IE_EXCEPTION << "Input has unexpected size " << input_bytes << ", expected " << graphDesc._inputDesc->totalSize;
    }

    ncStatus_t status;

    status = ncFifoWriteElem(graphDesc._inputFifoHandle, input_data, graphDesc._inputDesc, nullptr);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to write input to FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    status = ncGraphQueueInference(graphDesc._graphHandle, &graphDesc._inputFifoHandle, &graphDesc._outputFifoHandle);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to queue inference: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    if (result_data != nullptr && result_bytes != nullptr) {
        getResult(graphDesc, result_data, result_bytes);
    }
}

void MyriadExecutor::getResult(GraphDesc &graphDesc, void **result_data, size_t *result_bytes) {
    LOG_INFO("Graph result");
#ifdef NNLOG
    ALOGI("Graph result");
#endif
    ncStatus_t status;
    ncTensorDescriptor_t resDesc = {};
    void *userParam = nullptr;
    status = ncFifoReadElem(graphDesc._outputFifoHandle, result_data, &resDesc, &userParam);
    if (status != NC_OK) {
        THROW_IE_EXCEPTION << "Failed to read output from FIFO: " << ncStatusToStr(graphDesc._graphHandle, status);
    }

    if (resDesc.totalSize != graphDesc._outputDesc->totalSize) {
        THROW_IE_EXCEPTION << "Output has unexpected size " << resDesc.totalSize << ", expected " << graphDesc._outputDesc->totalSize;
    }

    *result_bytes = resDesc.totalSize;
}

void MyriadExecutor::deallocateGraph(DevicePtr &device, GraphDesc &graphDesc) {

    LOG_INFO("MyriadExecutor::deallocateGraph");
    #ifdef NNLOG
    ALOGI("MyriadExecutor::deallocateGraph");
    #endif
      std::lock_guard<std::mutex> lock(device_mutex);
    if (device->_deviceHandle != nullptr) {
        if (graphDesc._inputFifoHandle != nullptr) {
            auto res = ncFifoDelete(graphDesc._inputFifoHandle);
            if (res != NC_OK)
                LOG_WARNING("ncFifoDelete result %s", ncStatusToStr(nullptr, res));
            graphDesc._inputFifoHandle = nullptr;
        }
        if (graphDesc._outputFifoHandle != nullptr) {
            auto res = ncFifoDelete(graphDesc._outputFifoHandle);
            if (res != NC_OK)
                LOG_WARNING("ncFifoDelete result %s", ncStatusToStr(nullptr, res));
            graphDesc._outputFifoHandle = nullptr;
        }
        if (graphDesc._graphHandle != nullptr) {
            auto res = ncGraphDeallocate(graphDesc._graphHandle);
            if (res !=NC_OK) {
                LOG_DEBUG("Deallocate Graph result %s.", ncStatusToStr(nullptr, res));

                #ifdef NNLOG
                ALOGI("Deallocate Graph result %s.", ncStatusToStr(nullptr, res));
                #endif
              }
            graphDesc._graphHandle = nullptr;
        }

        device->_executors -= 1;
    }
}

MyriadExecutor::~MyriadExecutor() {
}

const char *MyriadExecutor::ncStatusToStr(graphHandle_t *graphHandle, ncStatus_t status) {
#define MVNC_STATUS_TO_STR(E) case E: return #E;
    switch (status) {
        MVNC_STATUS_TO_STR(NC_OK)
        MVNC_STATUS_TO_STR(NC_BUSY)
        MVNC_STATUS_TO_STR(NC_ERROR)
        MVNC_STATUS_TO_STR(NC_OUT_OF_MEMORY)
        MVNC_STATUS_TO_STR(NC_DEVICE_NOT_FOUND)
        MVNC_STATUS_TO_STR(NC_INVALID_PARAMETERS)
        MVNC_STATUS_TO_STR(NC_TIMEOUT)
        MVNC_STATUS_TO_STR(NC_MVCMD_NOT_FOUND)
        MVNC_STATUS_TO_STR(NC_NOT_ALLOCATED)
        MVNC_STATUS_TO_STR(NC_UNAUTHORIZED)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_FEATURE)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_GRAPH_FILE)
        MVNC_STATUS_TO_STR(NC_UNSUPPORTED_CONFIGURATION_FILE)
        case NC_MYRIAD_ERROR: {
            if (graphHandle == nullptr) {
                return "NC_MYRIAD_ERROR";
            } else {
                auto debugInfo = getGraphInfo<char>(graphHandle, NC_OPTION_CLASS0, NC_RO_GRAPH_DEBUG_INFO);
                if (debugInfo->numElements() == 0) {
                    return "NC_MYRIAD_ERROR";
                } else {
                    return debugInfo->info();
                }
            }
        }
        default:
            return "UNKNOWN MVNC STATUS";
    }
#undef MVNC_STATUS_TO_STR
}

void MyriadExecutor::printThrottlingStatus() {
// TODO: enable when needed
#if 0
    unsigned int thermal_stats_len = 0;
    float* thermal_stats;
    auto status = ncDeviceGetOption(_deviceHandle, NC_OPTION_CLASS0,
                                    NC_RO_DEVICE_THERMAL_STATS,
                                    reinterpret_cast<void **>(&thermal_stats),
                                    &thermal_stats_len);
    int throttling = static_cast<int>(thermal_stats[0]);

    if (status != NC_OK) {
        LOG_WARNING("WARNING: Failed to get Throttling and Thermal information with error: %s",
                    ncStatusToStr(nullptr, status));
    } else {
        if (throttling == 0) {
            LOG_INFO("** Device %d temperature normal (%.1lf C) **",
                     _deviceIdx, thermal_stats[1]);
        } else if (throttling == 1) {
            LOG_INFO("** Device %d temperature high (%.1lf C) - thermal throttling initiated **",
                     _deviceIdx, thermal_stats[1]);
        } else if (throttling == 2) {
            LOG_WARNING("*********************** WARNING *************************\n"\
                        "  Device %d temperature critical (%.1lf C)\n"               \
                        "  Aggressive thermal throttling initiated\n"                \
                        "  Continued use may result in device damage\n"              \
                        "*********************************************************",
                        _deviceIdx, thermal_stats[1]);
        }
    }
#endif
}

std::shared_ptr<GraphInfo<float>> MyriadExecutor::getPerfTimeInfo(graphHandle_t *graphHandle) {
    return getGraphInfo<float>(graphHandle, NC_OPTION_CLASS0, NC_RO_GRAPH_TIME_TAKEN);
}
