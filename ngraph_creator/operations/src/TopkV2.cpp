#include <TopkV2.hpp>
#undef LOG_TAG
#define LOG_TAG "TopkV2"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

TopkV2::TopkV2(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

void TopkV2::connectOperationToGraph() { createNode(); }

std::shared_ptr<ov::Node> TopkV2::createNode() {
    // Creating input nodes
    std::shared_ptr<ov::Node> input;

    input = getInputNode(0);

    auto k = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    int axis = -1;  // to find largest entries for the last dimension.

    auto k_node = createConstNode(ov::element::i32, {}, convertToVector(k));
    const auto topk = std::make_shared<ov::opset3::TopK>(
        input, k_node, axis, ov::opset3::TopK::Mode::MAX, ov::opset3::TopK::SortType::SORT_VALUES);

    auto outputNode = topk->outputs();

    for (int i = 0; i < 2; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        // TODO: remove this dummy convert
        std::shared_ptr<ov::Node> outNode;
        if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::f32);
        } else if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT16)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::f16);
        } else if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_INT32)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::i32);
        } else if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::f32);
            outNode = QuantizeNode(outNode, outputIndex, ov::element::u8);
        } else if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED)) {
            outNode = std::make_shared<ov::opset3::Convert>(outputNode[i], ov::element::f32);
            outNode = QuantizeNode(outNode, outputIndex, ov::element::i8);
        }

        mNgraphNodes->setOutputAtOperandIndex(outputIndex, outNode);
        ALOGD("%s Set Output index %d", __func__, outputIndex);
        const auto op = sModelInfo->getOperand(outputIndex);
        if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(outputIndex, outNode);
            ALOGD("%s Add result %d", __func__, outputIndex);
        }
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
