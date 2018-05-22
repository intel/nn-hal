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

#include "parsed_config.h"
#include <cpp_interfaces/exception2status.hpp>
#include <vector>

using namespace InferenceEngine;
using namespace VPU::Common;
using namespace InferenceEngine::VPUConfigParams;


namespace {

void parseStringList(const std::string &str, std::vector<std::string> &vec) {
    vec.clear();

    if (str.empty())
        return;

    std::istringstream istr(str);

    std::string elem;
    while (std::getline(istr, elem, ',')) {
        vec.push_back(elem);
    }
}

}  // namespace

LogLevel ParsedConfig::parseLogLevel(const std::string &option) {
    LogLevel logLevel = eLOGNONE;

    if (option.compare("LOG_WARNING") == 0) {
        logLevel = eLOGWARNING;
    } else if (option.compare("LOG_INFO") == 0) {
        logLevel = eLOGINFO;
    } else if (option.compare("LOG_DEBUG") == 0) {
        logLevel = eLOGDEBUG;
    }

    return logLevel;
}

ParsedConfig::ParsedConfig(const int platform, const std::map<std::string, std::string> &_config) {
    auto config = getDefaultConfig(platform);
    for (auto &option : _config) {
        config[option.first] = option.second;
    }
    validate(config, platform);

    blobConfig.firstShave = stoi(config[VPU_CONFIG_KEY(FIRST_SHAVE)]);
    blobConfig.lastShave = stoi(config[VPU_CONFIG_KEY(LAST_SHAVE)]);

    auto parseOptimizationOption = [](const std::string &option) {
        bool result = false;

        if (option.compare(CONFIG_VALUE(YES)) == 0) {
            result = true;
        } else if (option.compare(CONFIG_VALUE(NO)) == 0) {
            result = false;
        } else {
            THROW_IE_EXCEPTION << "Incorrect value for optimization option";
        }

        return result;
    };
    blobConfig.copyOptimization = parseOptimizationOption(config[VPU_CONFIG_KEY(COPY_OPTIMIZATION)]);
    blobConfig.reshapeOptimization = parseOptimizationOption(config[VPU_CONFIG_KEY(RESHAPE_OPTIMIZATION)]);
    blobConfig.memoryOptimization = parseOptimizationOption(config[VPU_CONFIG_KEY(MEMORY_OPTIMIZATION)]);
    blobConfig.ignoreUnknownLayers = parseOptimizationOption(config[VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS)]);
    blobConfig.hwOptimization = parseOptimizationOption(config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)]);
    blobConfig.useCmxBuffers = parseOptimizationOption(config[VPU_CONFIG_KEY(USE_CMX_BUFFERS)]);
    exclusiveAsyncRequests = parseOptimizationOption(config[CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)]);
    printReceiveTensorTime = parseOptimizationOption(config[VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME)]);

    blobConfig.cmxBufferStart = stoi(config[VPU_CONFIG_KEY(CMX_BUFFER_START)]);
    blobConfig.cmxBufferSize = stoi(config[VPU_CONFIG_KEY(CMX_BUFFER_SIZE)]);

    parseStringList(config[VPU_CONFIG_KEY(NONE_LAYERS)], blobConfig.NoneLayers);
    parseStringList(config[VPU_CONFIG_KEY(HW_WHITE_LIST)], blobConfig.hwWhiteList);
    parseStringList(config[VPU_CONFIG_KEY(HW_BLACK_LIST)], blobConfig.hwBlackList);

    float norm = stof(config[VPU_CONFIG_KEY(INPUT_NORM)]);
    blobConfig.inputScale = 1.f / norm;
    blobConfig.inputBias = stof(config[VPU_CONFIG_KEY(INPUT_BIAS)]);
}

void ParsedConfig::validate(const std::map<std::string, std::string> &_config, const int platform) {
    auto config = _config;
    auto defaultConfig = getDefaultConfig(MYRIAD_X);
    for (auto &option : config) {
        if (defaultConfig.find(option.first) == defaultConfig.end()) {
            THROW_IE_EXCEPTION << NOT_FOUND_str << option.first << " key is not exist";
        }
    }

    auto isLogLevel = [](const std::string &option) {
        return option.compare("LOG_NONE") == 0 || option.compare("LOG_WARNING") == 0
               || option.compare("LOG_INFO") == 0 || option.compare("LOG_DEBUG") == 0;
    };
    if (!isLogLevel(config[CONFIG_KEY(LOG_LEVEL)]) || !isLogLevel(config[VPU_CONFIG_KEY(LOG_LEVEL)]))
        THROW_IE_EXCEPTION << "Incorrect value for log option";

    if (platform == MYRIAD_2) {
        uint16_t firstShave = stoi(config[VPU_CONFIG_KEY(FIRST_SHAVE)]);
        if (firstShave > 11) {
            THROW_IE_EXCEPTION << "Incorrect value for KEY_VPU_FIRST_SHAVE option";
        }

        uint16_t lastShave = stoi(config[VPU_CONFIG_KEY(LAST_SHAVE)]);
        if (lastShave > 11 || firstShave > lastShave) {
            THROW_IE_EXCEPTION << "Incorrect value for KEY_VPU_LAST_SHAVE option";
        }
    } else {  // MYRIAD_X or UNKNOWN
        auto firstShaveValue = config[VPU_CONFIG_KEY(FIRST_SHAVE)];
        uint16_t firstShave = firstShaveValue.empty() ? 0 : stoi(firstShaveValue);
        if (firstShave > 15) {
            THROW_IE_EXCEPTION << "Incorrect value for KEY_VPU_FIRST_SHAVE option";
        }

        auto lastShaveValue = config[VPU_CONFIG_KEY(LAST_SHAVE)];
        uint16_t lastShave = lastShaveValue.empty() ? 15 : stoi(config[VPU_CONFIG_KEY(LAST_SHAVE)]);
        if (lastShave > 15 || firstShave > lastShave) {
            THROW_IE_EXCEPTION << "Incorrect value for KEY_VPU_LAST_SHAVE option";
        }
    }

    auto isOptimizationOption = [](const std::string &option) {
        return option.compare(CONFIG_VALUE(YES)) == 0 || option.compare(CONFIG_VALUE(NO)) == 0;
    };

    if (platform == MYRIAD_X) {
        if (!isOptimizationOption(config[VPU_CONFIG_KEY(COPY_OPTIMIZATION)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(RESHAPE_OPTIMIZATION)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(MEMORY_OPTIMIZATION)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(USE_CMX_BUFFERS)]) ||
            !isOptimizationOption(config[CONFIG_KEY(PERF_COUNT)]) ||
            !isOptimizationOption(config[CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)])) {
            THROW_IE_EXCEPTION << "Incorrect value for optimization option";
        }
    } else {  // MYRIAD_2 or UNKNOWN
        if (!isOptimizationOption(config[VPU_CONFIG_KEY(COPY_OPTIMIZATION)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(RESHAPE_OPTIMIZATION)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(MEMORY_OPTIMIZATION)]) ||
            !isOptimizationOption(config[VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS)]) ||
            !isOptimizationOption(config[CONFIG_KEY(PERF_COUNT)]) ||
            !isOptimizationOption(config[CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)])) {
           THROW_IE_EXCEPTION << "Incorrect value for optimization option";
       }
    }

    if (platform == MYRIAD_X) {
        uint32_t cmxBufferStart = stoi(config[VPU_CONFIG_KEY(CMX_BUFFER_START)]);
        uint32_t cmxBufferSize = stoi(config[VPU_CONFIG_KEY(CMX_BUFFER_SIZE)]);
        const size_t MAX_KEY_CMX_BUFFER_SIZE = 16 * 128 * 1024;
        if (cmxBufferStart + cmxBufferSize > MAX_KEY_CMX_BUFFER_SIZE) {
            THROW_IE_EXCEPTION << "[VPU] Incorrect value for CMX buffer range";
        }
    }

    float norm = stof(config[VPU_CONFIG_KEY(INPUT_NORM)]);
    if (norm == 0.0f) {
        THROW_IE_EXCEPTION << "Incorrect zero value for KEY_VPU_INPUT_NORM option";
    }
}

std::map<std::string, std::string> ParsedConfig::getDefaultConfig(const int platform) {
    if (platform == MYRIAD_X) {
        return {{VPU_CONFIG_KEY(FIRST_SHAVE),      "0"},
                {VPU_CONFIG_KEY(LAST_SHAVE),       "7"},
                {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),   CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(MEMORY_OPTIMIZATION),    CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(COPY_OPTIMIZATION),      CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(RESHAPE_OPTIMIZATION),   CONFIG_VALUE(YES)},
                {CONFIG_KEY(LOG_LEVEL),                  CONFIG_VALUE(LOG_NONE)},
                {VPU_CONFIG_KEY(LOG_LEVEL),              CONFIG_VALUE(LOG_NONE)},
                // vpu plugins ignore this key, they measure performance always,
                // added just to pass behavior tests
                {CONFIG_KEY(PERF_COUNT),                 CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(INPUT_NORM),       "1.0"},
                {VPU_CONFIG_KEY(INPUT_BIAS),       "0.0"},
                {VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS),  CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(NONE_LAYERS),      ""},
                {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(USE_CMX_BUFFERS),        CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(HW_WHITE_LIST),    ""},
                {VPU_CONFIG_KEY(HW_BLACK_LIST),    ""},
                {VPU_CONFIG_KEY(CMX_BUFFER_START), "0"},
                {VPU_CONFIG_KEY(CMX_BUFFER_SIZE),  "1048576"},
                {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME),    CONFIG_VALUE(NO)}
        };
    } else if (platform == MYRIAD_2) {
        return {{VPU_CONFIG_KEY(FIRST_SHAVE),      "0"},
                {VPU_CONFIG_KEY(LAST_SHAVE),       "11"},
                {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),   CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(MEMORY_OPTIMIZATION),    CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(COPY_OPTIMIZATION),      CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(RESHAPE_OPTIMIZATION),   CONFIG_VALUE(YES)},
                {CONFIG_KEY(LOG_LEVEL),                  CONFIG_VALUE(LOG_NONE)},
                {VPU_CONFIG_KEY(LOG_LEVEL),              CONFIG_VALUE(LOG_NONE)},
                // vpu plugins ignore this key, they measure performance always,
                // added just to pass behavior tests
                {CONFIG_KEY(PERF_COUNT),                 CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(INPUT_NORM),       "1.0"},
                {VPU_CONFIG_KEY(INPUT_BIAS),       "0.0"},
                {VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS),  CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(NONE_LAYERS),      ""},
                {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(USE_CMX_BUFFERS),        CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(HW_WHITE_LIST),    ""},
                {VPU_CONFIG_KEY(HW_BLACK_LIST),    ""},
                {VPU_CONFIG_KEY(CMX_BUFFER_START), "0"},
                {VPU_CONFIG_KEY(CMX_BUFFER_SIZE),  "0"},
                {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME),    CONFIG_VALUE(NO)}
        };
    } else {
        return {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),   CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(MEMORY_OPTIMIZATION),    CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(COPY_OPTIMIZATION),      CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(RESHAPE_OPTIMIZATION),   CONFIG_VALUE(YES)},
                {CONFIG_KEY(LOG_LEVEL),                  CONFIG_VALUE(LOG_NONE)},
                {VPU_CONFIG_KEY(LOG_LEVEL),              CONFIG_VALUE(LOG_NONE)},
                // vpu plugins ignore this key, they measure performance always,
                // added just to pass behavior tests
                {CONFIG_KEY(PERF_COUNT),                 CONFIG_VALUE(YES)},
                {VPU_CONFIG_KEY(INPUT_NORM),       "1.0"},
                {VPU_CONFIG_KEY(INPUT_BIAS),       "0.0"},
                {VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS),  CONFIG_VALUE(NO)},
                {VPU_CONFIG_KEY(NONE_LAYERS),      ""},
                {VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME),    CONFIG_VALUE(NO)}
        };
    }
}
