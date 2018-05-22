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
#include "xml_parse_utils.h"
#include "details/ie_exception.hpp"
#include "ie_precision.hpp"

int XMLParseUtils::GetIntAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    return atoi(attr.value());
}

std::string XMLParseUtils::GetStrAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    return attr.value();
}

std::string XMLParseUtils::GetStrAttr(const pugi::xml_node &node, const char *str, const char *def) {
    auto attr = node.attribute(str);
    if (attr.empty()) return def;
    return attr.value();
}

float XMLParseUtils::GetFloatAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    return static_cast<float>(atof(attr.value()));
}

InferenceEngine::Precision XMLParseUtils::GetPrecisionAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    return InferenceEngine::Precision::FromStr(attr.value());
}

InferenceEngine::Precision XMLParseUtils::GetPrecisionAttr(const pugi::xml_node &node, const char *str,
                                                           InferenceEngine::Precision def) {
    auto attr = node.attribute(str);
    if (attr.empty()) return InferenceEngine::Precision(def);
    return InferenceEngine::Precision::FromStr(attr.value());
}

int XMLParseUtils::GetIntAttr(const pugi::xml_node &node, const char *str, int defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    return atoi(attr.value());
}

float XMLParseUtils::GetFloatAttr(const pugi::xml_node &node, const char *str, float defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    return static_cast<float>(atof(attr.value()));
}

int XMLParseUtils::GetIntChild(const pugi::xml_node &node, const char *str, int defVal) {
    auto child = node.child(str);
    if (child.empty()) return defVal;
    return atoi(child.child_value());
}

std::string XMLParseUtils::NameFromFilePath(const char *filepath) {
    std::string baseName = filepath;
    auto slashPos = baseName.rfind('/');
    slashPos = slashPos == std::string::npos ? 0 : slashPos + 1;
    auto dotPos = baseName.rfind('.');
    if (dotPos != std::string::npos) {
        baseName = baseName.substr(slashPos, dotPos - slashPos);
    } else {
        baseName = baseName.substr(slashPos);
    }
    return baseName;
}

