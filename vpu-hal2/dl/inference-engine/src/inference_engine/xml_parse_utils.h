//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
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
//
#pragma once

#include <cstdlib>
#include <debug.h>
#include "pugixml.hpp"
#include "ie_common.h"
#include "ie_api.h"
#include <string>
#include <ie_precision.hpp>

#define FOREACH_CHILD(c, p, tag) for (auto c = p.child(tag); !c.empty(); c = c.next_sibling(tag))

namespace XMLParseUtils {

INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node &node, const char *str, int defVal);

INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node &node, const char *str, const char *def);

INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node &node, const char *str, float defVal);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision) GetPrecisionAttr(const pugi::xml_node &node, const char *str);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision)
GetPrecisionAttr(const pugi::xml_node &node, const char *str, InferenceEngine::Precision def);

INFERENCE_ENGINE_API_CPP(int) GetIntChild(const pugi::xml_node &node, const char *str, int defVal);

INFERENCE_ENGINE_API_CPP(std::string) NameFromFilePath(const char *filepath);

}  // namespace XMLParseUtils
