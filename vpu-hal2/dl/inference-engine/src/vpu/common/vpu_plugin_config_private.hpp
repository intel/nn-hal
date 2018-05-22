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

#pragma once

#include <string>
#include <vpu/vpu_plugin_config.hpp>

namespace InferenceEngine {
namespace VPUConfigParams {

DECLARE_VPU_CONFIG_KEY(FIRST_SHAVE);
DECLARE_VPU_CONFIG_KEY(LAST_SHAVE);

DECLARE_VPU_CONFIG_KEY(MEMORY_OPTIMIZATION);

DECLARE_VPU_CONFIG_KEY(COPY_OPTIMIZATION);

DECLARE_VPU_CONFIG_KEY(RESHAPE_OPTIMIZATION);

DECLARE_VPU_CONFIG_KEY(NONE_LAYERS);
DECLARE_VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS);

DECLARE_VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION);

DECLARE_VPU_CONFIG_KEY(USE_CMX_BUFFERS);
DECLARE_VPU_CONFIG_KEY(CMX_BUFFER_START);
DECLARE_VPU_CONFIG_KEY(CMX_BUFFER_SIZE);

DECLARE_VPU_CONFIG_KEY(HW_WHITE_LIST);
DECLARE_VPU_CONFIG_KEY(HW_BLACK_LIST);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
