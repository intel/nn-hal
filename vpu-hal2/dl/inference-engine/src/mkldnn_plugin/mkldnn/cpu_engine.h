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

#include "inference_engine.hpp"
#include "desc_layer.h"
#include "desc_tensor.h"
#include "desc_tensor_comb.h"

#include "cpu_prim_layer.h"
#include "cpu_prim_tensor.h"

#include "mkldnn.hpp"
#include <memory>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {
class CpuEngine;

using CpuEnginePtr = std::shared_ptr<CpuEngine>;

class CpuEngine : public details::no_copy {
public:
    CpuEngine() : eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0)) {}

    void bindThreads();

    void createDescription(DescTensorPtr tns, bool isWeights = false);

    void createDescription(DescLayerPtr layer);

    void setFlatFormat(DescTensorPtr tns);

    void createPrimitive(DescTensorPtr tns);

    void createPrimitive(DescLayerPtr tns);

    void setData(const TBlob<float> &src, DescTensorPtr dst);

    void getData(const DescTensorPtr src, TBlob<float> &dst);

    void subtraction(DescTensorPtr dst, DescTensorPtr sub);

    void subtraction(DescTensorPtr dst, std::vector<float> sub);

    void score(std::vector<DescLayerPtr> layers);

    void score(DescLayerPtr layer);

    void process(std::vector<mkldnn::primitive> exec_queue);

    mkldnn::engine eng;  // TODO: Move me back to private section

private:
    static inline mkldnn::memory::desc *get_desc(std::vector<DescTensorPtr> tensors, size_t indx = 0);

    static inline mkldnn::memory::desc *get_desc(DescTensorPtr tns);

    static inline mkldnn::memory *get_prim(std::vector<DescTensorPtr> tns, size_t indx = 0);

    static inline mkldnn::memory *get_prim(DescTensorPtr tns);

    void createPrimitiveCombined(DescTensorComb &tns, void *data);
};
}  // namespace MKLDNNPlugin