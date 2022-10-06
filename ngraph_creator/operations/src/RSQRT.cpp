#include <RSQRT.hpp>
#undef LOG_TAG
#define LOG_TAG "RSQRT"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

RSQRT::RSQRT(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

std::shared_ptr<ov::Node> RSQRT::createNode() {
    // Creating input nodes
    auto input = getInputNode(0);
    auto sqrtNode = std::make_shared<ov::opset3::Sqrt>(input);

    std::shared_ptr<ov::Node> constNode;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16))
        constNode = createConstNode(ov::element::f16, {1}, convertToVector(1.0));
    else
        constNode = createConstNode(ov::element::f32, {1}, convertToVector(1.0));

    auto outputNode = std::make_shared<ov::opset3::Divide>(constNode, sqrtNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
