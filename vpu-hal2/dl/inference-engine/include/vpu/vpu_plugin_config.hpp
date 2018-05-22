// Copyright (c) 2017-2018 Intel Corporation
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
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <string>
#include "../ie_plugin_config.hpp"

#define VPU_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_##name)
#define DECLARE_VPU_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_##name)
#define DECLARE_VPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_##name)

namespace InferenceEngine {
namespace VPUConfigParams {

/**
* @brief Desirable log level for devices.
* This option should be used with values: CONFIG_VALUE(LOG_NONE) (default),
* CONFIG_VALUE(LOG_WARNING), CONFIG_VALUE(LOG_INFO), CONFIG_VALUE(LOG_DEBUG)
*/
DECLARE_VPU_CONFIG_KEY(LOG_LEVEL);

/**
* @brief Normalization coefficient for the network input.
* This should be a real number.
*/
DECLARE_VPU_CONFIG_KEY(INPUT_NORM);

/**
* @brief Bias value that is added to each element of the network input.
* This should be a real number.
*/
DECLARE_VPU_CONFIG_KEY(INPUT_BIAS);

/**
* @brief Flag for adding to the profiling information the time of obtaining a tensor
* This option should be used with values: CONFIG_VALUE(NO) (default) or CONFIG_VALUE(YES)
*/
DECLARE_VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
