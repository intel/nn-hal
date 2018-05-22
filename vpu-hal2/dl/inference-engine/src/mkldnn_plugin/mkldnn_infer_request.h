//
// INTEL CONFIDENTIAL
// Copyright 2017 Intel Corporation.
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

#include "mkldnn_graph.h"
#include <memory>
#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

namespace MKLDNNPlugin {

class MKLDNNInferRequest : public InferenceEngine::InferRequestInternal {
public:
    typedef std::shared_ptr<MKLDNNInferRequest> Ptr;
    explicit MKLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs,
                          InferenceEngine::OutputsDataMap networkOutputs);

    void Infer() override;

    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;

    /**
     * @brief Given optional implementation of setting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) override;

    /**
     * @brief Given optional implementation of getting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) override;

    void SetGraph(const MKLDNNGraph::Ptr& graph);

private:
    template <typename T> void pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob);

    void changeAllPtrs(void *oldPtr, void *newPtr);
    void changeDefaultPtr();
    void resetDefaultPtr();
    MKLDNNGraph::Ptr graph;
    std::map<std::string, void*> externalPtr;
    std::map<std::string, void*> defaultPtr;
};
}  // namespace MKLDNNPlugin
