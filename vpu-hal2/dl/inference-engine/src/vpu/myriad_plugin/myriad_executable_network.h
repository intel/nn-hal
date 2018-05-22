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
#include <queue>
#include <sstream>
#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/ie_executor_manager.hpp>
#include "myriad_executor.h"
#include "myriad_executable_network.h"
#include "graph_transformer.hpp"
#include "myriad_infer_request.h"
#include <environment.h>
#include <parsed_config.h>
#include "myriad_async_infer_request.h"

namespace VPU {
namespace MyriadPlugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<ExecutableNetwork> Ptr;

    explicit ExecutableNetwork(InferenceEngine::ICNNNetwork &network,
                               std::vector<DevicePtr> &devicePool,
                               const std::map<std::string, std::string> &config) {
        Common::LogLevel logLevel;
        Common::LogLevel vpuLogLevel;

        try {
            logLevel = Common::ParsedConfig::parseLogLevel(
                    config.at(CONFIG_KEY(LOG_LEVEL)));
            vpuLogLevel = Common::ParsedConfig::parseLogLevel(
                    config.at(VPU_CONFIG_KEY(LOG_LEVEL)));
        } catch (const std::out_of_range& error) {
            auto default_config = Common::ParsedConfig::getDefaultConfig();
            logLevel = Common::ParsedConfig::parseLogLevel(
                    default_config.at(CONFIG_KEY(LOG_LEVEL)));
            vpuLogLevel = Common::ParsedConfig::parseLogLevel(
                    default_config.at(VPU_CONFIG_KEY(LOG_LEVEL)));
        }
        _log = std::make_shared<Common::Logger>();
        _log->init(logLevel);

        _executor = std::make_shared<MyriadExecutor>(vpuLogLevel, _log);
        _device = _executor->openDevice(devicePool);
        _env = std::make_shared<Common::Environment>(_device->_platform, config);
        // ignore hardware optimization config for MYRIAD2, it is always disabled
        if (_device->_platform == MYRIAD_2) {
            _env->parsedConfig.blobConfig.hwOptimization = false;
            LOG_INFO("[VPU] hardware optimization config for MYRIAD2 always disabled");
        }

        auto graphTrasnformer = createGraphTransformer(_env->parsedConfig.blobConfig, _log);

        size_t numStages = 0;
        graphTrasnformer->generate(network, _graphBlob, _env->blobMetaData, numStages);

        LOG_INFO("[VPU] ExecutableNetwork : graphTrasnformer->generate done");

        char networkName[1024] = {};
        network.getName(networkName, sizeof(networkName));
        LOG_INFO("[VPU] org network name %s", networkName);
        _executor->allocateGraph(_device, _graphDesc, _graphBlob, numStages, networkName);
        LOG_INFO("[VPU] _executor->allocateGraph");
        if (_env->parsedConfig.exclusiveAsyncRequests) {
            InferenceEngine::ExecutorManager *executorManager = InferenceEngine::ExecutorManager::getInstance();
            _taskExecutor = executorManager->getExecutor(
                    InferenceEngine::TargetDeviceInfo::name(InferenceEngine::TargetDevice::eMYRIAD));
        }

        for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
            std::stringstream idStream;
            idStream << networkName << "_TaskExecutorGetResult" << i;
            _taskExecutorGetResultIds.push(idStream.str());
        }
    }

    ~ExecutableNetwork() {
        _executor->deallocateGraph(_device, _graphDesc);
    }

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<MyriadInferRequest>(_graphDesc, networkInputs, networkOutputs, _env, _log, _executor);
    }

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override {
        auto syncRequestImpl = std::make_shared<MyriadInferRequest>(_graphDesc, _networkInputs, _networkOutputs, _env, _log,
                                                                    _executor);
        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        auto taskExecutorGetResult = getNextTaskExecutotGetResult();
        auto asyncTreadSafeImpl = std::make_shared<MyriadAsyncInferRequest>(
                syncRequestImpl, _taskExecutor, taskExecutorGetResult, _taskSynchronizer, _callbackExecutor);
        asyncRequest.reset(new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(
                           asyncTreadSafeImpl),
                           [](InferenceEngine::IInferRequest *p) { p->Release(); });
        asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    }

    void Export(const std::string &modelFileName) {
        THROW_IE_EXCEPTION << "Export is not implemented\n";
    }

    void GetMappedTopology(
            std::map<std::string, std::vector<InferenceEngine::PrimitiveInfo::Ptr>> &deployedTopology) override {
        THROW_IE_EXCEPTION << "GetMappedTopology is not implemented\n";
    }

private:
    Common::EnvironmentPtr _env;
    Common::LoggerPtr _log;
    MyriadExecutorPtr _executor;
    std::vector<char> _graphBlob;
    GraphDesc _graphDesc;
    DevicePtr _device;

    const size_t _maxTaskExecutorGetResultCount = 1;
    std::queue<std::string> _taskExecutorGetResultIds;

    InferenceEngine::ITaskExecutor::Ptr getNextTaskExecutotGetResult() {
        std::string id = _taskExecutorGetResultIds.front();

        _taskExecutorGetResultIds.pop();
        _taskExecutorGetResultIds.push(id);

        InferenceEngine::ExecutorManager *executorManager = InferenceEngine::ExecutorManager::getInstance();
        InferenceEngine::ITaskExecutor::Ptr taskExecutor = executorManager->getExecutor(id);

        return taskExecutor;
    }
};

}  // namespace MyriadPlugin
}  // namespace VPU
