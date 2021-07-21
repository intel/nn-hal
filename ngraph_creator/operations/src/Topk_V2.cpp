#include <Topk_V2.hpp>
#define LOG_TAG "Topk_V2"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Topk_V2::Topk_V2(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Topk_V2::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    return true;
}

void Topk_V2::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> Topk_V2::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    auto k = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    int axis = -1;  // to find largest entries for the last dimension.

    auto k_node = createConstNode(ngraph::element::i32, {}, convertToVector(k));
    const auto topk =
        std::make_shared<ngraph::opset3::TopK>(input, k_node, axis, ngraph::opset3::TopK::Mode::MAX,
                                               ngraph::opset3::TopK::SortType::SORT_VALUES);

    auto outputNode = topk->outputs();

    for (int i = 0; i < 2; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        // TODO: remove this dummy convert
        std::shared_ptr<ngraph::Node> outNode;
        if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) {
            outNode =
                std::make_shared<ngraph::opset3::Convert>(outputNode[i], ngraph::element::f32);
        } else if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_INT32)) {
            outNode =
                std::make_shared<ngraph::opset3::Convert>(outputNode[i], ngraph::element::i32);
        } else if (checkOutputOperandType(i, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
            outNode =
                std::make_shared<ngraph::opset3::Convert>(outputNode[i], ngraph::element::f32);
            outNode = QuantizeNode(outNode, outputIndex, ngraph::element::u8);
        }

        mNgraphNodes->setOutputAtOperandIndex(outputIndex, outNode);
        ALOGD("%s Set Output index %d", __func__, outputIndex);
        const auto op = sModelInfo->getOperand(outputIndex);
        if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
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
