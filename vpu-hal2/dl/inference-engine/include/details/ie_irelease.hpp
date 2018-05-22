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

/**
 * @brief A header file for the Inference Engine plugins destruction mechanism
 * @file ie_irelease.hpp
 */
#pragma once

#include "ie_no_copy.hpp"
#include <memory>

namespace InferenceEngine {
namespace details {
/**
 * @class IRelease
 * @brief This class is used for objects allocated by a shared module (in *.so)
 */
class IRelease : public no_copy {
public:
    /**
     * @brief Releases current allocated object and all related resources.
     * Once this method is called, the pointer to this interface is no longer valid
     */
    virtual void Release() noexcept = 0;

 protected:
    /**
     * @brief Default destructor
     */
    ~IRelease() override = default;
};



template <class T> inline std::shared_ptr<T> shared_from_irelease(T * ptr) {
    std::shared_ptr<T> pointer(ptr, [](IRelease *p) {
        p->Release();
    });
    return pointer;
}

}  // namespace details
}  // namespace InferenceEngine
