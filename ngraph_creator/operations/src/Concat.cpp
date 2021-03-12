#include <Concat.hpp>
#define LOG_TAG "Concat"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Concat::Concat(const Operation& op) : OperationsBase(op) {
    mDefaultOutputIndex = mNnapiOp.outputs[0];
}

bool Concat::validate() { return true; }

std::shared_ptr<ngraph::Node> Concat::createNode() {
    auto n = mNnapiOp.inputs.size() - 1;                    // 0 ~ n-1: The list of n input tensors
    auto axis = ParseOperationInput(*sModel, mNnapiOp, n);  // n: concatenation axis
    std::vector<ngraph::Output<ngraph::Node>> inputs;
    ALOGD("createNode n %d, axis %d", n, axis);
    for (int i = 0; i < n; i++) {
        auto inputIndex = mNnapiOp.inputs[i];
        auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
        const auto op = sModel->operands[inputIndex];
        ALOGD("createNode inputIndex %d, lifetime %d", inputIndex, op.lifetime);
        if (mNgraphNodes->isForcedNchw(inputIndex)) {
            inputOp = transpose(NCHW_NHWC, inputOp);
            mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
        }
        inputs.push_back(inputOp);
    }

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Concat>(inputs, axis);
    const auto op = sModel->operands[mDefaultOutputIndex];
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
