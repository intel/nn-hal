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

#include <vector>
#include <string>
#include <w_unistd.h>
#include <w_dirent.h>
#include <debug.h>
#include <algorithm>
#include <file_utils.h>

#include "mkldnn_extension_mngr.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::MKLDNNPlugin;

IMKLDNNGenericPrimitive* MKLDNNExtensionManager::CreateExtensionPrimitive(const CNNLayerPtr& layer) {
    IMKLDNNGenericPrimitive* primitive = nullptr;

    // last registered has a priority
    for (auto ext = _extensions.rbegin(); ext != _extensions.rend(); ++ext) {
        ResponseDesc respDesc;
        StatusCode rc;
        auto *mkldnnExtension = dynamic_cast<IMKLDNNExtension *>(ext->get());
        if (mkldnnExtension != nullptr) {
            // If extension does not want to provide impl it should just return OK and do nothing.
            rc = mkldnnExtension->CreateGenericPrimitive(primitive, layer, &respDesc);
            if (rc != OK) {
                primitive = nullptr;
                continue;
            }

            if (primitive != nullptr) {
                break;
            }
        }
    }
    return primitive;
}

void MKLDNNExtensionManager::AddExtension(IExtensionPtr extension) {
    _extensions.push_back(extension);
}

InferenceEngine::ILayerImplFactory* MKLDNNExtensionManager::CreateExtensionFactory(
        const InferenceEngine::CNNLayerPtr &layer) {
    if (!layer)
        THROW_IE_EXCEPTION << "Cannot get cnn layer!";
    ILayerImplFactory* factory = nullptr;
    for (auto& ext : _extensions) {
        ResponseDesc responseDesc;
        StatusCode rc;
        rc = ext->getFactoryFor(factory, layer.get(), &responseDesc);
        if (rc != OK) {
            factory = nullptr;
            continue;
        }
        if (factory != nullptr) {
            break;
        }
    }
    return factory;
}



