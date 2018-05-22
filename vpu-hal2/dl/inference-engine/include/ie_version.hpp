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
 * @brief A header file that provides versioning information for the inference engine shared library
 * @file ie_version.hpp
 */
#pragma once

#include "ie_api.h"

namespace InferenceEngine {

/**
 * @struct Version
 * @brief  Represents version information that describes plugins and the inference engine runtime library
 */
#pragma pack(push, 1)
struct Version {
    /**
     * @brief An API version reflects the set of supported features
     */
    struct {
        int major;
        int minor;
    } apiVersion;
    /**
     * @brief A build number
     */
    const char * buildNumber;
    /**
     * @brief A null terminated description string
     */
    const char * description;
};
#pragma pack(pop)

/**
 * @brief Gets the current Inference Engine version
 * @return The current Inference Engine version
 */
INFERENCE_ENGINE_API(const Version*) GetInferenceEngineVersion() noexcept;

}  // namespace InferenceEngine
