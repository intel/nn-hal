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
    ALOGD("createNode n %d, axis %d", n, axis);
    for (int i = 0; i < n; i++) {
        auto inputIndex = operation.inputs[i];
        auto inputOp = mNgraphNodes->getOperationOutput(inputIndex);
        const auto op = mModel.operands[inputIndex];
        ALOGD("createNode inputIndex %d, lifetime %d", inputIndex, op.lifetime);
        if (op.lifetime == OperandLifeTime::CONSTANT_COPY ||
            op.lifetime == OperandLifeTime::CONSTANT_REFERENCE ||
            op.lifetime ==
                OperandLifeTime::MODEL_INPUT)  // TODO: should use NNAPI_Utils::isConst || isInput
        {
            inputOp = transpose(NHWC_NCHW, inputOp);
        }
        inputs.push_back(inputOp);
    }

    std::shared_ptr<ngraph::Node> outputNode =
        std::make_shared<ngraph::opset3::Concat>(inputs, axis);
    const auto outputIndex = operation.outputs[0];
    const auto op = mModel.operands[outputIndex];
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        mNgraphNodes->setResultNode(outputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
