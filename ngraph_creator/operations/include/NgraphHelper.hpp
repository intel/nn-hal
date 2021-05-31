#pragma once
#include <log/log.h>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
using FusedActivationFunc = V1_0::FusedActivationFunc;

static inline ngraph::Shape toNgraphShape(const std::vector<uint32_t>& dimensions) {
    ngraph::Shape shapeVec;
    for (size_t i = 0; i < dimensions.size(); i++) {
        shapeVec.push_back(static_cast<size_t>(dimensions[i]));
    }
    return shapeVec;
}

static inline std::shared_ptr<ngraph::Node> applyActivation(std::shared_ptr<ngraph::Node> inputNode,
                                                            int32_t activationFn) {
    std::shared_ptr<ngraph::Node> activationNode = nullptr;
    switch (activationFn) {
        case (int32_t)FusedActivationFunc::NONE:
            ALOGV("Adding No Activation");
            return inputNode;
            break;
        case (int32_t)FusedActivationFunc::RELU:
            ALOGV("Adding relu");
            activationNode = std::make_shared<ngraph::opset3::Relu>(inputNode);
            break;
        case (int32_t)FusedActivationFunc::RELU6:
            ALOGV("Adding relu6");
            activationNode = std::make_shared<ngraph::opset3::Clamp>(inputNode, 0, 6);
            break;
        case (int32_t)FusedActivationFunc::RELU1:
            ALOGV("Adding relu1");
            activationNode = std::make_shared<ngraph::opset3::Clamp>(inputNode, -1, 1);
            break;
        default:
            ALOGI("UNKNOWN ACTIVATION FUNCTION %d !!!!!", activationFn);
            return inputNode;
    }
    return activationNode;
}

static inline void calculateExplicitPadding(int32_t in_size, int32_t stride, int32_t filter_size,
                                            int32_t padding_implicit, int32_t* padding_head,
                                            int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == 1) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android