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
 * @brief A header that defines advanced related properties for CPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods
 *
 * @file ie_helpers.hpp
 */


#include "IRDocument.h"
#include "IRLayers.h"
#include "IENetwork.h"
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>

#include "ie_iinfer_request.hpp"
#include "ie_infer_request.hpp"
#include "ie_plugin_cpp.hpp"
#include "ie_exception_conversion.hpp"
#include "ie_builders.hpp"
#include "ie_network.hpp"
//#include "debug.h"
#include <fstream>

#include <android/log.h>
#include <log/log.h>

#ifdef ENABLE_MYRIAD
#include "vpu_plugin_config.hpp"
#endif
using namespace InferenceEngine::details;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

}
}
}
}