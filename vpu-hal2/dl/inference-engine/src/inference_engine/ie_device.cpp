/*
* INTEL CONFIDENTIAL
* Copyright 2016 Intel Corporation.
*
* The source code contained or described herein and all documents
* related to the source code ("Material") are owned by Intel Corporation
* or its suppliers or licensors. Title to the Material remains with
* Intel Corporation or its suppliers and licensors. The Material may
* contain trade secrets and proprietary and confidential information
* of Intel Corporation and its suppliers and licensors, and is protected
* by worldwide copyright and trade secret laws and treaty provisions.
* No part of the Material may be used, copied, reproduced, modified,
* published, uploaded, posted, transmitted, distributed, or disclosed
* in any way without Intel's prior express written permission.
*
* No license under any patent, copyright, trade secret or other
* intellectual property right is granted to or conferred upon you by
* disclosure or delivery of the Materials, either expressly, by implication,
* inducement, estoppel or otherwise. Any license under such intellectual
* property rights must be express and approved by Intel in writing.
*
* Include any supplier copyright notices as supplier requires Intel to use.
*
* Include supplier trademarks or logos as supplier requires Intel to use,
* preceded by an asterisk. An asterisked footnote can be added as follows:
* *Third Party trademarks are the property of their respective owners.
*
* Unless otherwise agreed by Intel in writing, you may not remove or alter
* this notice or any other notice embedded in Materials by Intel or Intel's
* suppliers or licensors in any way.
*/

#include <ie_device.hpp>
#include <details/ie_exception.hpp>
#include "description_buffer.hpp"

using namespace InferenceEngine;

FindPluginResponse InferenceEngine::findPlugin(const FindPluginRequest& req) {
    switch (req.device) {
    case TargetDevice::eCPU:
        return { {
#ifdef ENABLE_MKL_DNN
                "MKLDNNPlugin",
#endif
#ifdef ENABLE_OPENVX_CVE
                "OpenVXPluginCVE",
#elif defined ENABLE_OPENVX
                "OpenVXPlugin",
#endif
            } };
    case TargetDevice::eGPU:
        return { {
#ifdef ENABLE_CLDNN
                "clDNNPlugin",
#endif
#ifdef ENABLE_OPENVX
                "OpenVXPlugin",
#endif
            } };
    case TargetDevice::eFPGA:
        return{ {
#ifdef ENABLE_DLIA
                "dliaPlugin",
#endif
#ifdef ENABLE_OPENVX
                "OpenVXPlugin",
#endif
            } };
    case TargetDevice::eMYRIAD:
        return{ {
#ifdef ENABLE_MYRIAD
                "myriadPlugin",
#endif
            } };

        case TargetDevice::eGNA:
            return{ {
#ifdef ENABLE_GNA
                        "GNAPlugin",
#endif
                    } };
    case TargetDevice::eHETERO:
        return{ {
                "HeteroPlugin",
            } };

    default:
        THROW_IE_EXCEPTION << "Cannot find plugin for device: " << getDeviceName(req.device);
    }
}

INFERENCE_ENGINE_API(StatusCode) InferenceEngine::findPlugin(
        const FindPluginRequest& req, FindPluginResponse& result, ResponseDesc* resp) noexcept {
    try {
        result = findPlugin(req);
    }
    catch (const std::exception& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    }
    return OK;
}
