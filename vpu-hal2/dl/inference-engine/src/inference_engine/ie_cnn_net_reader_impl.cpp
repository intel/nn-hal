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
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <map>

#include "debug.h"
#include "parsers.h"
#include "ie_cnn_net_reader_impl.h"
#include "v2_format_parser.h"
#include <file_utils.h>
#include <ie_plugin.hpp>
#include "xml_parse_utils.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::string CNNNetReaderImpl::NameFromFilePath(const char* filepath) {
    string modelName = filepath;
    auto slashPos = modelName.rfind('/');
    slashPos = slashPos == std::string::npos ? 0 : slashPos + 1;
    auto dotPos = modelName.rfind('.');
    if (dotPos != std::string::npos) {
        modelName = modelName.substr(slashPos, dotPos - slashPos);
    } else {
        modelName = modelName.substr(slashPos);
    }
    return modelName;
}

CNNNetReaderImpl::CNNNetReaderImpl() : parseSuccess(false), version(0) {
}

StatusCode CNNNetReaderImpl::SetWeights(const TBlob<uint8_t>::Ptr& weights, ResponseDesc* desc)  noexcept {
    if (!_parser) {
        return DescriptionBuffer(desc) << "network must be read first";
    }
    try {
        _parser->SetWeights(weights);
    }
    catch (const InferenceEngineException& iee) {
        return DescriptionBuffer(desc) << iee.what();
    }

    return OK;
}

int CNNNetReaderImpl::GetFileVersion(pugi::xml_node& root) {
    return XMLParseUtils::GetIntAttr(root, "version", 0);
}

StatusCode CNNNetReaderImpl::ReadNetwork(const void* model, size_t size, ResponseDesc* resp) noexcept {
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_buffer(model, size);
    if (res.status != pugi::status_ok) {
        return DescriptionBuffer(resp) << res.description() << "at offset " << res.offset;
    }
    StatusCode ret = ReadNetwork(xmlDoc);
    if (ret != OK) {
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

StatusCode CNNNetReaderImpl::ReadWeights(const char* filepath, ResponseDesc* resp) noexcept {
    long long fileSize = FileUtils::fileSize(filepath);
    if (fileSize == 0)
        return OK;
    if (fileSize < 0)
        return DescriptionBuffer(resp) << "filesize for: " << filepath << " - " << fileSize << "<0";

    if (network.get() == nullptr) {
        return DescriptionBuffer(resp) << "network is empty";
    }

    size_t ulFileSize = static_cast<size_t>(fileSize);

    TBlob<uint8_t>::Ptr weightsPtr(new TBlob<uint8_t>(Precision::U8, C, {ulFileSize}));
    weightsPtr->allocate();
    try {
        FileUtils::readAllFile(filepath, weightsPtr->buffer(), ulFileSize);
    }
    catch (const InferenceEngineException& iee) {
        return DescriptionBuffer(resp) << iee.what();
    }

    return SetWeights(weightsPtr, resp);
}

StatusCode CNNNetReaderImpl::ReadNetwork(const char* filepath, ResponseDesc* resp) noexcept {
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_file(filepath);
    if (res.status != pugi::status_ok) {
        std::ifstream t(filepath);
        std::string str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());

        int line = 1;
        int pos = 0;
        for (auto token : str) {
            if (token == '\n') {
                line++;
                pos = 0;
            } else {
                pos++;
            }
            if (pos >= res.offset) {
                break;
            }
        }

        return DescriptionBuffer(resp) << "Error loading xmlfile: " << filepath << ", " << res.description()
                                       << " at line: " << line << " pos: " << pos;
    }
    StatusCode ret = ReadNetwork(xmlDoc);
    if (ret != OK) {
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

StatusCode CNNNetReaderImpl::ReadNetwork(pugi::xml_document& xmlDoc) {
    description.clear();

    try {
        // check which version it is...
        pugi::xml_node root = xmlDoc.document_element();

        version = GetFileVersion(root);
        if (version > 2) THROW_IE_EXCEPTION << "cannot parse future versions: " << version;
        _parser.reset(new details::V2FormatParser(version));
        network = _parser->Parse(root);
        name = network->getName();

        parseSuccess = true;
    } catch (const std::string& err) {
        description = err;
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (const InferenceEngineException& e) {
        description = e.what();
        parseSuccess = false;
        return GENERAL_ERROR;
    }

    return OK;
}

INFERENCE_ENGINE_API(InferenceEngine::ICNNNetReader*) InferenceEngine::CreateCNNNetReader() noexcept {
    return new CNNNetReaderImpl;
}
