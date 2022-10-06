#include <Split.hpp>
#undef LOG_TAG
#define LOG_TAG "Split"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Split::Split(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

void Split::connectOperationToGraph() { createNode(); }

std::shared_ptr<ov::Node> Split::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> splitNode;

    splitNode = getInputNode(0, false);

    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axisNode = createConstNode(ov::element::i32, {}, convertToVector(axis));
    auto numSplits = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);

    auto outputNode =
        std::make_shared<ov::opset3::Split>(splitNode, axisNode, numSplits)->outputs();

    for (size_t i = 0; i < numSplits; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        // TODO: remove this dummy convert
        std::shared_ptr<ov::Node> outNode;
        if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::f32);
        } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::f16);
        } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::i32);
        } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) ||
                   checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::u8);
        }

        // auto outNode = outputNode[i].get_node_shared_ptr();
        mNgraphNodes->setOutputAtOperandIndex(outputIndex, outNode);
        const auto op = sModelInfo->getOperand(outputIndex);
        if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(outputIndex, outNode);
        }
    }

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
