#include <Concat.hpp>
#define LOG_TAG "Concat"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Concat::Concat(const Model& model) : OperationsBase(model) {}

bool Concat::validate(const Operation& op) { return true; }

std::shared_ptr<ngraph::Node> Concat::createNode(const Operation& operation) {
    auto n = operation.inputs.size() - 1; //0 ~ n-1: The list of n input tensors
    auto axis = ParseOperationInput(mModel, operation, n); //n: concatenation axis
    std::vector<ngraph::Output<ngraph::Node>> inputs;
    const auto outputIndex = operation.outputs[0];
    ALOGD("createNode n %d, axis %d", n, axis);
    for (int i = 0; i < n; i++) {
        auto inputIndex = operation.inputs[i];
        auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
        const auto op = mModel.operands[inputIndex];
        ALOGD("createNode inputIndex %d, lifetime %d", inputIndex, op.lifetime);
        if(mNgraphNodes->isForcedNchw(inputIndex)) {
            inputOp = transpose(NCHW_NHWC, inputOp);
            mNgraphNodes->setForcedNchw(outputIndex, false);
        }
        inputs.push_back(inputOp);
    }

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Concat>(inputs, axis);
    const auto op = mModel.operands[outputIndex];
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(outputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
