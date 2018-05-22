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

/**
 * \brief inference engine executanle network API wrapper, to be used by particular implementors
 * \file ie_executable_network_base.hpp
 */

#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "cpp_interfaces/exception2status.hpp"

namespace InferenceEngine {

/**
 * @brief cpp interface for executable network, to avoid dll boundaries and simplify internal development
 * @tparam T Minimal CPP implementation of IExecutableNetwork (e.g. ExecutableNetworkInternal)
 */
template<class T>
class ExecutableNetworkBase : public IExecutableNetwork {
    std::shared_ptr<T> _impl;

public:
    typedef std::shared_ptr<ExecutableNetworkBase<T>> Ptr;

    explicit ExecutableNetworkBase(std::shared_ptr<T> impl) {
        if (impl.get() == nullptr) {
            THROW_IE_EXCEPTION << "implementation not defined";
        }
        _impl = impl;
    }

    StatusCode CreateInferRequest(IInferRequest::Ptr &req, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->CreateInferRequest(req));
    }

    StatusCode Export(const std::string &modelFileName, ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->Export(modelFileName));
    }

    StatusCode GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology,
                                 ResponseDesc *resp) noexcept override {
        TO_STATUS(_impl->GetMappedTopology(deployedTopology));
    }

    void Release() noexcept override {
        delete this;
    }

    // Need for unit tests only
    const std::shared_ptr<T> getImpl() const {
        return _impl;
    }

private:
    ~ExecutableNetworkBase() = default;
};

}  // namespace InferenceEngine
