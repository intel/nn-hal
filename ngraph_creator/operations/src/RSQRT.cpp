#include <RSQRT.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

RSQRT::RSQRT(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool RSQRT::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> RSQRT::createNode() {
    // Creating input nodes
    auto input = getInputNode<float>(0);

    auto sqrtNode = std::make_shared<ngraph::opset3::Sqrt>(input);
    auto constNode =
        ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0});
    auto outputNode = std::make_shared<ngraph::opset3::Divide>(constNode, sqrtNode);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
