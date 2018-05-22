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

#include <ie_api.h>
#include <memory>
#include "mkldnn_memory.h"
#include "mkldnn_dims.h"
#include <map>

namespace MKLDNNPlugin {

class MKLDNNNode;
class MKLDNNEdge;

using MKLDNNEdgePtr = std::shared_ptr<MKLDNNEdge>;
using MKLDNNEdgeWeakPtr = std::weak_ptr<MKLDNNEdge>;

class MKLDNNEdge : public InferenceEngine::details::no_copy {
public:
    enum class Status {
        Uninitialized,
        NeedAllocation,
        NotAllocated,
        Allocated,
        Validated
    };
    MKLDNNEdge(const std::shared_ptr<MKLDNNNode>& parent, const std::shared_ptr<MKLDNNNode>& child);

    inline Status getStatus() noexcept {
        return status;
    }

    void changeStatus(Status state);

    virtual void allocate();
    virtual void validate();

    const std::shared_ptr<MKLDNNNode> getParent() const;
    const std::shared_ptr<MKLDNNNode> getChild() const;

    bool needReorder();

    virtual const MKLDNNMemory& getMemory();

    virtual MKLDNNMemoryPtr& getMemoryPtr();

    inline bool contains(const std::shared_ptr<MKLDNNNode>& node) const noexcept {
        return node.get() == parent.lock().get() || node.get() == child.lock().get();
    }

    bool operator==(const MKLDNNEdge& edge) {
        return this->parent.lock().get() == edge.parent.lock().get() &&
                this->child.lock().get() == edge.child.lock().get();
    }

    bool isDropped();

    MKLDNNMemoryDesc getInputDesc();
    int getInputNum();
    MKLDNNMemoryDesc getOutputDesc();
    int getOutputNum();

    MKLDNNDims &getDims();
    void setDims(MKLDNNDims &dims);

    void sharedMemFrom(const MKLDNNEdgePtr& edge);
    MKLDNNEdgePtr getSharedEdge() const;


private:
    std::weak_ptr<MKLDNNNode> parent;
    std::weak_ptr<MKLDNNNode> child;
    MKLDNNEdgeWeakPtr memoryFromEdge;
    MKLDNNDims dims;
    MKLDNNMemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    mkldnn::memory::format getSpecifiedInputFormat(std::map<mkldnn::memory::format, size_t> formats);
    mkldnn::memory::format getSpecifiedOutputFormat(std::map<mkldnn::memory::format, size_t> formats);
    MKLDNNMemoryDesc inputDesc;
    MKLDNNMemoryDesc outputDesc;

    bool nodeCanChangeDesc(const std::shared_ptr<MKLDNNPlugin::MKLDNNNode>& node) const;
};

}  // namespace MKLDNNPlugin
