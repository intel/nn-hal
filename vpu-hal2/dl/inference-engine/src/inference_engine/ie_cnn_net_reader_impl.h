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

#include "ie_icnn_net_reader.h"
#include "cnn_network_impl.hpp"
#include <memory>
#include <string>
#include <map>

namespace pugi {
class xml_node;

class xml_document;
}  // namespace pugi

namespace InferenceEngine {
namespace details {
class CNNNetReaderImpl : public ICNNNetReader {
public:
    static std::string NameFromFilePath(const char *filepath);

    CNNNetReaderImpl();

    StatusCode ReadNetwork(const char *filepath, ResponseDesc *resp) noexcept override;

    StatusCode ReadNetwork(const void *model, size_t size, ResponseDesc *resp)noexcept override;

    StatusCode SetWeights(const TBlob<uint8_t>::Ptr &weights, ResponseDesc *resp) noexcept override;

    StatusCode ReadWeights(const char *filepath, ResponseDesc *resp) noexcept override;

    ICNNNetwork *getNetwork(ResponseDesc *resp) noexcept override {
        return network.get();
    }


    bool isParseSuccess(ResponseDesc *resp) noexcept override {
        return parseSuccess;
    }


    StatusCode getDescription(ResponseDesc *desc) noexcept override {
        return DescriptionBuffer(OK, desc) << description;
    }


    StatusCode getName(char *name, size_t len, ResponseDesc *resp) noexcept override {
        strncpy(name, this->name.c_str(), len - 1);
        if (len) name[len-1] = '\0';  // strncpy is not doing this, so output might be not null-terminated
        return OK;
    }

    int getVersion(ResponseDesc * resp) noexcept override {
        return version;
    }

    void Release() noexcept override {
        delete this;
    }

private:
    std::shared_ptr<InferenceEngine::details::IFormatParser> _parser;

    static int GetFileVersion(pugi::xml_node &root);

    StatusCode ReadNetwork(pugi::xml_document &xmlDoc);

    std::string description;
    std::string name;
    InferenceEngine::details::CNNNetworkImplPtr network;
    bool parseSuccess;
    int version;
};
}  // namespace details
}  // namespace InferenceEngine
