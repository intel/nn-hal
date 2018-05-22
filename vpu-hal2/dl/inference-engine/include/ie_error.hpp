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
 * @brief A header file for a plugin logging mechanism
 * @file ie_error.hpp
 */
#pragma once

namespace InferenceEngine {
/**
 * @class IErrorListener
 * @brief This class represents a custom error listener.
 * Plugin consumers can provide it via InferenceEngine::SetLogCallback
 */
class IErrorListener {
public:
    /**
     * @brief The plugin calls this method with a null terminated error message (in case of error)
     * @param msg Error message
     */
    virtual void onError(const char *msg) noexcept = 0;
};
}  // namespace InferenceEngine
