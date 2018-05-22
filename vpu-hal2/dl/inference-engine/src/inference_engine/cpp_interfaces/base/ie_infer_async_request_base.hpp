// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <map>
#include <string>
#include "ie_iinfer_request.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "ie_profiling.hpp"

namespace InferenceEngine {

/**
 * @brief cpp interface for async infer request, to avoid dll boundaries and simplify internal development
 * @tparam T Minimal CPP implementation of IInferRequest (e.g. AsyncInferRequestThreadSafeDefault)
 */
template<class T>
class InferRequestBase : public IInferRequest {
protected:
    std::shared_ptr<T> _impl;

public:
    typedef std::shared_ptr<InferRequestBase<T>> Ptr;

    explicit InferRequestBase(std::shared_ptr<T> impl) : _impl(impl) {}

    StatusCode Infer(ResponseDesc *resp) noexcept override {
        IE_PROFILING_AUTO_SCOPE(Infer);
        TO_STATUS(_impl->Infer());
    }

    StatusCode GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap,
                                    ResponseDesc *resp) const noexcept override {
        TO_STATUS(_impl->GetPerformanceCounts(perfMap));
    }

    StatusCode SetBlob(const char *name, const Blob::Ptr &data, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->SetBlob(name, data));
    }

    StatusCode GetBlob(const char *name, Blob::Ptr &data, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->GetBlob(name, data));
    }

    StatusCode StartAsync(ResponseDesc *resp) noexcept override {
        IE_PROFILING_AUTO_SCOPE(StartAsync);
        TO_STATUS(_impl->StartAsync());
    }

    StatusCode Wait(int64_t millis_timeout, ResponseDesc *resp) noexcept override {
        IE_PROFILING_AUTO_SCOPE(Wait);
        NO_EXCEPT_CALL_RETURN_STATUS(_impl->Wait(millis_timeout));
    }

    StatusCode SetCompletionCallback(CompletionCallback callback) noexcept override {
        TO_STATUS_NO_RESP(_impl->SetCompletionCallback(callback));
    }

    StatusCode GetUserData(void **data, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->GetUserData(data));
    }

    StatusCode SetUserData(void *data, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->SetUserData(data));
    }

    void Release() noexcept override {
        delete this;
    }

protected:
    ~InferRequestBase() = default;
};

}  // namespace InferenceEngine
