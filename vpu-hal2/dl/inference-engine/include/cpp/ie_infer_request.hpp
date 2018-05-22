// Copyright (c) 2018 Intel Corporation
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
#include <string>
#include <map>
#include "ie_iinfer_request.hpp"
#include "details/ie_exception_conversion.hpp"

#ifdef NNLOG
#include <android/log.h>
#include <cutils/log.h>
#endif

namespace InferenceEngine {

class ICompletionCallbackWrapper {
public:
    virtual ~ICompletionCallbackWrapper() = default;

    virtual void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept = 0;
};

template<class T>
class CompletionCallbackWrapper : public ICompletionCallbackWrapper {
    T lambda;
public:
    explicit CompletionCallbackWrapper(const T &lambda) : lambda(lambda) {}

    void call(InferenceEngine::IInferRequest::Ptr /*request*/,
              InferenceEngine::StatusCode /*code*/) const noexcept override {
        lambda();
    }
};

template<>
class CompletionCallbackWrapper<IInferRequest::CompletionCallback> : public ICompletionCallbackWrapper {
    IInferRequest::CompletionCallback callBack;
public:
    explicit CompletionCallbackWrapper(const IInferRequest::CompletionCallback &callBack) : callBack(callBack) {}

    void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept override {
        callBack(request, code);
    }
};

/**
 * @class InferRequest
 * @note This class is a wrapper of IInferRequest to provide setters/getters of input/output which operates with BlobMap's
 */
class InferRequest {
    IInferRequest::Ptr actual;
    std::shared_ptr<ICompletionCallbackWrapper> callback;

    static void callWrapper(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) {
        ICompletionCallbackWrapper *pWrapper = nullptr;
        ResponseDesc dsc;
        request->GetUserData(reinterpret_cast<void**>(&pWrapper), &dsc);
        pWrapper->call(request, code);
    }

public:
    InferRequest() = default;

    /**
     * @brief Sets input/output data to infer
     * @note: Memory allocation does not happen
     * @param name Name of input or output blob.
     * @param data Reference to input or output blob. The type of a blob must match the network input precision and size.
     */
    void SetBlob(const std::string &name, const Blob::Ptr &data) {
      #ifdef NNLOG
      ALOGI("ALOG InferRequest SetBlob");
      #endif
        CALL_STATUS_FNC(SetBlob, name.c_str(), data);
    }

    /**
     * @brief Gets input/output data for inference
     * @note: Memory allocation does not happen
     * @param name Name of input or output blob.
     * @throws Status code if operation failed
     * @return Shared pointer to input or output blob. The type of Blob is same as network input or output precision respectively.
     */
    Blob::Ptr GetBlob(const std::string &name) {
        Blob::Ptr data;
        CALL_STATUS_FNC(GetBlob, name.c_str(), data);
        return data;
    }

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of IInferRequest while request is ongoing (running or waiting in queue)
     * @throws Status code if operation failed
     */
    void Infer() {
        CALL_STATUS_FNC_NO_ARGS(Infer);
    }

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  @note: not all plugins may provide meaningful data
     * @throws Status code if operation failed
     */
    std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const {
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        CALL_STATUS_FNC(GetPerformanceCounts, perfMap);
        return perfMap;
    }

    /**
    * @brief Set input data to infer
    * @note: Memory allocation doesn't happen
    * @param inputs - a reference to a map of input blobs accessed by input names. The type of Blob must correspond to the network input precision and size.
    */
    void SetInput(const BlobMap &inputs) {
        for (auto &&input : inputs) {
            CALL_STATUS_FNC(SetBlob, input.first.c_str(), input.second);
        }
    }

    /**
     * @brief Set data that will contain result of the inference
     * @note: Memory allocation doesn't happen
     * @param results - a reference to a map of result blobs accessed by output names. The type of Blob must correspond to the network output precision and size.
     */
    void SetOutput(const BlobMap &results) {
        for (auto &&result : results) {
            CALL_STATUS_FNC(SetBlob, result.first.c_str(), result.second);
        }
    }

    /**
     * constructs InferRequest from initialised shared_pointer
     * @param actual
     */
    explicit InferRequest(IInferRequest::Ptr request) : actual(request) {}

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     * @note: It returns immediately. Inference starts also immediately.
     */
    void StartAsync() {
        CALL_STATUS_FNC_NO_ARGS(StartAsync);
    }

    /**
     * @brief Waits for the result to become available. Blocks until specified millis_timeout has elapsed or the result becomes available, whichever comes first.
     * @param millis_timeout - maximum duration in milliseconds to block for
     * @note There are special cases when millis_timeout is equal some value of WaitMode enum:
     * * STATUS_ONLY - immediately returns request status (IInferRequest::RequestStatus). It doesn't block or interrupt current thread.
     * * RESULT_READY - waits until inference result becomes available
     * @return Enumeration of the resulted action: OK (0) for success
     */
    StatusCode Wait(int64_t millis_timeout) {
        return actual->Wait(millis_timeout, nullptr);
    }

    template <class T>
    void SetCompletionCallback(const T & callbackToSet) {
        callback.reset(new CompletionCallbackWrapper<T>(callbackToSet));
        CALL_STATUS_FNC(SetUserData, callback.get());
        actual->SetCompletionCallback(callWrapper);
    }

    /**
     * @brief  IInferRequest pointer to be used directly in CreateInferRequest functions
     */
    operator IInferRequest::Ptr &() {
        return actual;
    }

    bool operator!() const noexcept {
        return !actual;
    }

    explicit operator bool() const noexcept {
        return !!actual;
    }

    typedef std::shared_ptr<InferRequest> Ptr;
};

}  // namespace InferenceEngine
