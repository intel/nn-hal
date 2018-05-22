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

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_async_only.hpp>
#include <graph_transformer.hpp>

#include "hddl_executor.h"
#include "hddl_infer_request.h"

#include "environment.h"
#include "parsed_config.h"
#include "hddl_allocator.h"

namespace VPU {
namespace HDDLPlugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeAsyncOnly {
public:
    explicit ExecutableNetwork(InferenceEngine::ICNNNetwork &network,
                               const std::map<std::string, std::string> &config,
                               HDDLAllocatorPtr &hddlAllocatorPtr) : _hddlAllocatorPtr(hddlAllocatorPtr) {
        _env = std::make_shared<Common::Environment>(MYRIAD_X, config);

        Common::LogLevel logLevel;
        try {
            logLevel = Common::ParsedConfig::parseLogLevel(
                    config.at(CONFIG_KEY(LOG_LEVEL)));
        } catch (const std::out_of_range& error) {
            auto default_config = Common::ParsedConfig::getDefaultConfig();
            logLevel = Common::ParsedConfig::parseLogLevel(
                    default_config.at(CONFIG_KEY(LOG_LEVEL)));
        }

        _log = std::make_shared<Common::Logger>();
        _log->init(logLevel);

        _executor = std::make_shared<Executor>(_log);
        _executor->openDevice();

        auto graphTrasnformer = createGraphTransformer(_env->parsedConfig.blobConfig, _log);

        size_t numStages = 0;
        graphTrasnformer->generate(network, _graphBlob, _env->blobMetaData, numStages);

        char networkName[1024] = {};
        network.getName(networkName, sizeof(networkName));
        _executor->allocateGraph(_graphBlob, numStages, networkName);
    }

    InferenceEngine::AsyncInferRequestInternal::Ptr
    CreateAsyncInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<HDDLInferRequest>(networkInputs, networkOutputs, _env, _log, _executor, _hddlAllocatorPtr);
    }

    virtual ~ExecutableNetwork() = default;

    void Export(const std::string &modelFileName) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void GetMappedTopology(
            std::map<std::string, std::vector<InferenceEngine::PrimitiveInfo::Ptr>> &deployedTopology) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

private:
    Common::EnvironmentPtr _env;
    Common::LoggerPtr _log;
    ExecutorPtr _executor;
    std::vector<char> _graphBlob;
    HDDLAllocatorPtr _hddlAllocatorPtr;
};

}  // namespace HDDLPlugin
}  // namespace VPU
