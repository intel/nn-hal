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

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <ie_common.h>
#include "myriad_executor.h"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include <environment.h>
#include <vpu_logger.h>

namespace VPU {
namespace MyriadPlugin {

class MyriadInferRequest : public InferenceEngine::InferRequestInternal {
    MyriadExecutorPtr _executor;
    Common::EnvironmentPtr _env;
    InferenceEngine::Layout _deviceLayout;
    Common::LoggerPtr _log;

    GraphDesc _graphDesc;

public:
    typedef std::shared_ptr<MyriadInferRequest> Ptr;

    explicit MyriadInferRequest(GraphDesc &_graphDesc, InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs,
                          const Common::EnvironmentPtr &env,
                          const Common::LoggerPtr &log,
                          const MyriadExecutorPtr &executor);

    void Infer() override;
    void InferAsync();
    void GetResult();

    void
    GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;
};

}  // namespace MyriadPlugin
}  // namespace VPU

