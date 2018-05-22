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

#include <debug.h>
#include "v2_format_parser.h"
#include "xml_parse_utils.h"
#include "range_iterator.hpp"
#include <vector>
#include <string>

inline pugi::xml_node GetChild(const pugi::xml_node& node, std::vector<std::string> tags, bool failIfMissing = true) {
    for (auto tag : tags) {
        pugi::xml_node dn = node.child(tag.c_str());
        if (!dn.empty()) return dn;
    }
    if (failIfMissing)
        THROW_IE_EXCEPTION << "missing <" << InferenceEngine::details::dumpVec(tags)
                           << "> Tags at offset :" << node.offset_debug();
    return pugi::xml_node();
}

using namespace XMLParseUtils;

namespace InferenceEngine {
namespace details {
template<class LT>
class V2LayerCreator : public BaseCreator {
public:
    explicit V2LayerCreator(const std::string& type) : BaseCreator(type) {}

    CNNLayer* CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) override {
        auto res = new LT(layerParsePrms.prms);

        // First parse for the generic representation
        pugi::xml_node dn = GetChild(node, { "data", tolower(res->type) + "_data" }, false);
        if (!dn.empty()) {
            for (auto ait = dn.attributes_begin(); ait != dn.attributes_end(); ++ait) {
                pugi::xml_attribute attr = *ait;
                res->params[attr.name()] = attr.value();
            }
        }

        // Then parse by specific layer type
        ParseNode(res, node);
        return res;
    }
private:
    static void ParseNode(LT* pLayer, pugi::xml_node& node);
};

std::vector<BaseCreator *> V2FormatParser::getCreators() const {
    static std::vector<BaseCreator *> creators = {
            new V2LayerCreator<PowerLayer>("Power"),
            new V2LayerCreator<ConvolutionLayer>("Convolution"),
            new V2LayerCreator<DeconvolutionLayer>("Deconvolution"),
            new V2LayerCreator<PoolingLayer>("Pooling"),
            new V2LayerCreator<FullyConnectedLayer>("InnerProduct"),
            new V2LayerCreator<FullyConnectedLayer>("FullyConnected"),
            new V2LayerCreator<NormLayer>("LRN"),
            new V2LayerCreator<NormLayer>("Norm"),
            new V2LayerCreator<SoftMaxLayer>("Softmax"),
            new V2LayerCreator<SoftMaxLayer>("SoftMax"),
            new V2LayerCreator<ReLULayer>("ReLU"),
#ifdef AKS
            new V2LayerCreator<TanHLayer>("TanH"),
            new V2LayerCreator<SigmoidLayer>("Sigmoid"),

#endif
            new V2LayerCreator<ClampLayer>("Clamp"),
            new V2LayerCreator<SplitLayer>("Split"),
            new V2LayerCreator<SplitLayer>("Slice"),
            new V2LayerCreator<ConcatLayer>("Concat"),
            new V2LayerCreator<EltwiseLayer>("Eltwise"),
            new V2LayerCreator<ScaleShiftLayer>("ScaleShift"),
            new V2LayerCreator<CropLayer>("Crop"),
            new V2LayerCreator<ReshapeLayer>("Reshape"),
            new V2LayerCreator<TileLayer>("Tile"),
            new V2LayerCreator<BatchNormalizationLayer>("BatchNormalization"),
    };
    return creators;
}

template<>
void V2LayerCreator<CNNLayer>::ParseNode(InferenceEngine::CNNLayer* pLayer, pugi::xml_node& node) {
    // Nothing to do
    pugi::xml_node dn = GetChild(node, { "data", tolower(pLayer->type) + "_data" }, false);
    if (dn.empty()) return;
    for (auto ait = dn.attributes_begin(); ait != dn.attributes_end(); ++ait) {
        pugi::xml_attribute attr = *ait;
        pLayer->params[attr.name()] = attr.value();
    }
}

template<>
void V2LayerCreator<PowerLayer>::ParseNode(InferenceEngine::PowerLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "power", "power_data", "data" });
    pLayer->offset = GetFloatAttr(dn, "shift");
    pLayer->power = GetFloatAttr(dn, "power");
    pLayer->scale = GetFloatAttr(dn, "scale");
}

template<>
void V2LayerCreator<ConvolutionLayer>::ParseNode(ConvolutionLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "convolution", "convolution_data", "data" });

    pLayer->_out_depth = XMLParseUtils::GetIntAttr(dn, "output");
    pLayer->_kernel_x = XMLParseUtils::GetIntAttr(dn, "kernel-x");
    pLayer->_kernel_y = XMLParseUtils::GetIntAttr(dn, "kernel-y");
    pLayer->_stride_x = XMLParseUtils::GetIntAttr(dn, "stride-x", 1);
    pLayer->_stride_y = XMLParseUtils::GetIntAttr(dn, "stride-y", 1);
    pLayer->_padding_x = XMLParseUtils::GetIntAttr(dn, "pad-x", 0);
    pLayer->_padding_y = XMLParseUtils::GetIntAttr(dn, "pad-y", 0);
    pLayer->_dilation_x = XMLParseUtils::GetIntAttr(dn, "dilation-x", 1);
    pLayer->_dilation_y = XMLParseUtils::GetIntAttr(dn, "dilation-y", 1);
    pLayer->_group = XMLParseUtils::GetIntAttr(dn, "group", 1);

    if (0 == pLayer->_stride_x) {
        pLayer->_stride_x = 1;
        LogError("Warning! in layer %s: Stride x is 0, setting to 1 (at XML offset %d)", pLayer->name.c_str(),
            node.offset_debug());
    }
    if (0 == pLayer->_stride_y) {
        pLayer->_stride_y = 1;
        LogError("Warning! in layer %s: Stride y is 0, setting to 1 (at XML offset %d)", pLayer->name.c_str(),
            node.offset_debug());
    }
}

template<>
void V2LayerCreator<DeconvolutionLayer>::ParseNode(DeconvolutionLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "deconvolution", "deconvolution_data", "data" });

    pLayer->_out_depth = XMLParseUtils::GetIntAttr(dn, "output");
    pLayer->_kernel_x = XMLParseUtils::GetIntAttr(dn, "kernel-x");
    pLayer->_kernel_y = XMLParseUtils::GetIntAttr(dn, "kernel-y");
    pLayer->_stride_x = XMLParseUtils::GetIntAttr(dn, "stride-x", 1);
    pLayer->_stride_y = XMLParseUtils::GetIntAttr(dn, "stride-y", 1);
    pLayer->_padding_x = XMLParseUtils::GetIntAttr(dn, "pad-x", 0);
    pLayer->_padding_y = XMLParseUtils::GetIntAttr(dn, "pad-y", 0);
    pLayer->_group = XMLParseUtils::GetIntAttr(dn, "group", 1);
    pLayer->_dilation_x = XMLParseUtils::GetIntAttr(dn, "dilation-x", 1);
    pLayer->_dilation_y = XMLParseUtils::GetIntAttr(dn, "dilation-y", 1);

    if (0 == pLayer->_stride_x) {
        pLayer->_stride_x = 1;
        LogError("Warning! in layer %s: Stride x is 0, setting to 1 (at XML offset %d)", pLayer->name.c_str(),
            node.offset_debug());
    }
    if (0 == pLayer->_stride_y) {
        pLayer->_stride_y = 1;
        LogError("Warning! in layer %s: Stride y is 0, setting to 1 (at XML offset %d)", pLayer->name.c_str(),
            node.offset_debug());
    }
}

template<>
void V2LayerCreator<PoolingLayer>::ParseNode(PoolingLayer* pLayer, pugi::xml_node& node) {
    // <PoolingData KernelX="3" KernelY="3" PadX="0" PadY="0" StrideX="2"
    // StrideY="2" RoundingType="Floor" PoolMethod="Max"/>
    // TODO: Support assimetric paramas + pool Method
    pugi::xml_node dn = GetChild(node, { "pooling", "pooling_data", "data" });

    int kernel_x = GetIntAttr(dn, "kernel-x", -1);
    /** Pooling as custom layer */
    if (kernel_x == -1) {
        int kernel_size = GetIntAttr(dn, "kernel_size");
        int kernel_w = GetIntAttr(dn, "kernel_w", 0);
        int kernel_h = GetIntAttr(dn, "kernel_h", 0);
        pLayer->_kernel_x = kernel_w == 0 ? kernel_size : kernel_w;
        pLayer->_kernel_y = kernel_h == 0 ? kernel_size : kernel_h;

        int stride = GetIntAttr(dn, "stride", 1);
        int stride_w = GetIntAttr(dn, "stride_w", 0);
        int stride_h = GetIntAttr(dn, "stride_h", 0);
        pLayer->_stride_x = stride_w == 0 ? stride : stride_w;
        pLayer->_stride_y = stride_h == 0 ? stride : stride_h;

        int pad = GetIntAttr(dn, "pad", 0);
        int pad_w = GetIntAttr(dn, "pad_w", 0);
        int pad_h = GetIntAttr(dn, "pad_h", 0);
        pLayer->_padding_x = pad_w == 0 ? pad : pad_w;
        pLayer->_padding_y = pad_h == 0 ? pad : pad_h;

        std::string alg = GetStrAttr(dn, "pool", "caffe.PoolingParameter.MAX");
        pLayer->_type = alg == "caffe.PoolingParameter.MAX" ? PoolingLayer::MAX : PoolingLayer::AVG;
    } else /** Default behaviour */ {
        pLayer->_kernel_x = GetIntAttr(dn, "kernel-x");
        pLayer->_kernel_y = GetIntAttr(dn, "kernel-y");
        pLayer->_stride_x = XMLParseUtils::GetIntAttr(dn, "stride-x", 1);
        pLayer->_stride_y = XMLParseUtils::GetIntAttr(dn, "stride-y", 1);
        pLayer->_padding_x = XMLParseUtils::GetIntAttr(dn, "pad-x", 0);
        pLayer->_padding_y = XMLParseUtils::GetIntAttr(dn, "pad-y", 0);

        // TODO: All kind of pool methods
        std::string excPad = XMLParseUtils::GetStrAttr(dn, "exclude-pad", "false");
        pLayer->_exclude_pad = excPad == "true" ? true : false;
        std::string alg = GetStrAttr(dn, "pool-method", "Max");
        pLayer->_type = alg == "avg" ? PoolingLayer::AVG : PoolingLayer::MAX;
    }
}

template<>
void V2LayerCreator<FullyConnectedLayer>::ParseNode(FullyConnectedLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "fc", "fc_data", "data" });
    pLayer->_out_num = GetIntAttr(dn, "out-size");
}


template<>
void V2LayerCreator<NormLayer>::ParseNode(NormLayer* pLayer, pugi::xml_node& node) {
    // <norm Alpha="0.0063999998" Beta="0.75" LocalSize="5" Region="Across"/>
    pugi::xml_node dn = GetChild(node, { "lrn", "norm", "norm_data", "data" });

    pLayer->_size = GetIntAttr(dn, "local-size", 0) + GetIntAttr(dn, "local_size", 0);
    pLayer->_k = GetIntAttr(dn, "k", 1);
    pLayer->_alpha = GetFloatAttr(dn, "alpha");
    pLayer->_beta = GetFloatAttr(dn, "beta");
    auto attr = dn.attribute("region");

    bool isSame = std::equal(null_terminated_string(attr.value()),
                                      null_terminated_string_end(),
                                      null_terminated_string("same"),
                                      [] (char c1, char c2) {
                                          return std::tolower(c1) == c2;
                                      });
    pLayer->_isAcrossMaps = attr.empty() || !isSame;
}

template<>
void V2LayerCreator<ConcatLayer>::ParseNode(ConcatLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "concat", "concat_data", "data" }, false);
    if (dn.empty()) return;
    pLayer->_axis = GetIntAttr(dn, "axis", 1);
}

template<>
void V2LayerCreator<EltwiseLayer>::ParseNode(EltwiseLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "elementwise", "elementwise_data", "data" }, false);

    if (dn.empty()) {
        return;
    }

    std::string op = GetStrAttr(dn, "operation", "");

    if (op == "sum") {
        pLayer->_operation = EltwiseLayer::Sum;
    } else if (op == "mul" || op == "prod") {
        pLayer->_operation = EltwiseLayer::Prod;
    } else if (op == "max") {
        pLayer->_operation = EltwiseLayer::Max;
    } else {
        THROW_IE_EXCEPTION << "Unsupported element wise operation: " << op;
    }

    auto getArray = [](std::string param, std::vector<float>& array) {
        std::istringstream stream(param);
        std::string str;
        while (getline(stream, str, ',')) {
            float val = std::stof(str);
            array.push_back(val);
        }
    };
    getArray(GetStrAttr(dn, "coeff", ""), pLayer->coeff);
}

template<>
void V2LayerCreator<CropLayer>::ParseNode(CropLayer *pLayer, pugi::xml_node &node) {
    pugi::xml_node dn = GetChild(node, { "crop", "crop-data", "data" }, false);

    if (dn.empty()) {
        return;
    }

    FOREACH_CHILD(_cn, dn, "crop") {
        int axis = GetIntAttr(_cn, "axis", 0);
        if (version_ == 1) axis++;
        pLayer->axis.push_back(axis);
        int offset = GetIntAttr(_cn, "offset", 0);
        pLayer->offset.push_back(offset);
        int dim = GetIntAttr(_cn, "dim", 0);
        pLayer->dim.push_back(dim);
    }

    if (pLayer->axis.size() == 0) {
        auto getArray = [](std::string param, std::vector<int>& array) {
            std::istringstream stream(param);
            std::string str;
            while (getline(stream, str, ',')) {
                int val = std::stoi(str);
                array.push_back(val);
            }
        };
        getArray(GetStrAttr(dn, "axis", ""), pLayer->axis);
        getArray(GetStrAttr(dn, "offset", ""), pLayer->offset);
        getArray(GetStrAttr(dn, "dim", ""), pLayer->dim);
    }
}

template<>
void V2LayerCreator<ReshapeLayer>::ParseNode(ReshapeLayer *pLayer, pugi::xml_node &node) {
    pugi::xml_node dn = GetChild(node, { "reshape_data", "data" }, false);

    if (dn.empty()) {
        return;
    }

    // TODO: Do we need to parse params as resolveOutput
    // is not used anymore?
    try {
        std::string dims = GetStrAttr(dn, "dim");

        std::istringstream stream(dims);
        std::string str;
        while (getline(stream, str, ',')) {
            int val = std::stoi(str);
            pLayer->shape.push_back(val);
        }

        pLayer->axis = GetIntAttr(dn, "axis");
        pLayer->num_axes = GetIntAttr(dn, "num_axes");
    } catch (...) {
        // Try to parse the first version
        FOREACH_CHILD(_cn, dn, "dim") {
            const pugi::char_t* dimVal = _cn.child_value();
            int dim = std::stoi(dimVal);
            pLayer->shape.push_back(dim);
        }
    }
}

template<>
void V2LayerCreator<TileLayer>::ParseNode(TileLayer *pLayer, pugi::xml_node &node) {
    pugi::xml_node dn = GetChild(node, { "tile_data", "data" }, false);

    if (dn.empty()) {
        return;
    }

    pLayer->axis = GetIntAttr(dn, "axis", -1);
    pLayer->tiles = GetIntAttr(dn, "tiles", -1);
}

template<>
void V2LayerCreator<SoftMaxLayer>::ParseNode(SoftMaxLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "softmax_data", "data" }, false);

    // Default value
    pLayer->axis = 1;

    if (dn.empty()) {
        return;
    }

    pLayer->axis = GetIntAttr(dn, "axis", 1);
}

template<>
void V2LayerCreator<ReLULayer>::ParseNode(ReLULayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "relu_data", "data" }, false);
    if (dn.empty()) return;
    pLayer->negative_slope = GetFloatAttr(dn, "negative_slope");
}
#ifdef AKS
template<>
void V2LayerCreator<TanHLayer>::ParseNode(TanHLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "tanh_data", "data" }, false);
    if (dn.empty()) return;
    pLayer->negative_slope = GetFloatAttr(dn, "negative_slope");
}
template<>
void V2LayerCreator<SigmoidLayer>::ParseNode(SigmoidLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "sigmoid_data", "data" }, false);
    if (dn.empty()) return;
    pLayer->negative_slope = GetFloatAttr(dn, "negative_slope");
}
#endif
template<>
void V2LayerCreator<ClampLayer>::ParseNode(ClampLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "clamp_data", "data" }, false);
    if (dn.empty()) return;
    pLayer->min_value = GetFloatAttr(dn, "min");
    pLayer->max_value = GetFloatAttr(dn, "max");
}

template<>
void V2LayerCreator<SplitLayer>::ParseNode(SplitLayer* pLayer, pugi::xml_node& node) {
    pugi::xml_node dn = GetChild(node, { "data", "split_data" }, false);
    if (dn.empty()) return;
    pLayer->_axis = GetIntAttr(dn, "axis", 1);
}

template<>
void V2LayerCreator<ScaleShiftLayer>::ParseNode(ScaleShiftLayer *pLayer, pugi::xml_node &node) {
    pugi::xml_node dn = GetChild(node, { "scale_shift", "scale_shift_data", "data" }, false);
    if (dn.empty()) return;
    pLayer->_broadcast = GetIntAttr(dn, "broadcast", 2);
}

template<>
void V2LayerCreator<BatchNormalizationLayer>::ParseNode(BatchNormalizationLayer *pLayer, pugi::xml_node &node) {
    pugi::xml_node dn = GetChild(node, { "batch_norm", "batch_norm_data", "data" }, false);
    if (dn.empty()) return;
    pLayer->epsilon = GetFloatAttr(dn, "epsilon");
}

}  // namespace details
}  // namespace InferenceEngine

/***********************************************************************************/
/******* End of Layer Parsers ******************************************************/
/***********************************************************************************/
