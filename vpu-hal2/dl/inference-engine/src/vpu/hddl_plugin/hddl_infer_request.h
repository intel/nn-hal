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
#include <ie_iinfer_request.hpp>
#include <environment.h>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_internal.hpp>

#include "hddl_executor.h"

namespace VPU {
namespace HDDLPlugin {

class HDDLInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeInternal,
                         public InferenceEngine::AsyncInferRequestInternal {
    ExecutorPtr _executor;
    Common::EnvironmentPtr _env;
    HDDLAllocatorPtr _hddlAllocatorPtr;
    uint64_t _taskHandle;

    InferenceEngine::BlobMap _hddlInputs;
    InferenceEngine::BlobMap _hddlOutputs;
    InferenceEngine::Layout _deviceLayout;
    Common::LoggerPtr _log;

public:
    explicit HDDLInferRequest(InferenceEngine::InputsDataMap networkInputs,
                              InferenceEngine::OutputsDataMap networkOutputs,
                              const Common::EnvironmentPtr &env,
                              const Common::LoggerPtr &log,
                              const ExecutorPtr &executor,
                              HDDLAllocatorPtr &hddlAllocatorPtr);

    virtual ~HDDLInferRequest();

    void GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) override {
        InferenceEngine::AsyncInferRequestThreadSafeInternal::GetBlob(name, data);
    }

    void SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) override {
        InferenceEngine::AsyncInferRequestThreadSafeInternal::SetBlob(name, data);
    }

    void StartAsync() override {
        InferenceEngine::AsyncInferRequestThreadSafeInternal::StartAsync();
    }

    void Infer() override {
        InferenceEngine::AsyncInferRequestThreadSafeInternal::Infer();
    }

    void SetCompletionCallback(InferenceEngine::IInferRequest::CompletionCallback callback) override {
        InferenceEngine::AsyncInferRequestThreadSafeInternal::SetCompletionCallback(callback);
    }

    void GetUserData(void **data) override {
        InferenceEngine::AsyncInferRequestThreadSafeInternal::GetUserData(data);
    }

    void SetUserData(void *data) override {
        InferenceEngine::AsyncInferRequestThreadSafeInternal::SetUserData(data);
    }

    void StartAsync_ThreadUnsafe() override;

    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override;

    void Infer_ThreadUnsafe() override;

    void
    GetPerformanceCounts_ThreadUnsafe(
            std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;

    void GetBlob_ThreadUnsafe(const char *name, InferenceEngine::Blob::Ptr &data) override;

    void SetBlob_ThreadUnsafe(const char *name, const InferenceEngine::Blob::Ptr &data) override;

    void SetCompletionCallback_ThreadUnsafe(InferenceEngine::IInferRequest::CompletionCallback callback) override;

    void GetUserData_ThreadUnsafe(void **data) override;

    void SetUserData_ThreadUnsafe(void *data) override;

    void *HDDLCallback();

private:
    void CopyToExternalOutputs();
};

using HDDLInferRequestPtr = std::shared_ptr<HDDLInferRequest>;

}  // namespace HDDLPlugin
}  // namespace VPU
