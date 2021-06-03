#include <Topk_V2.hpp>

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

    if (!checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        return false;
    }

    return true;
}

void Topk_V2::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> Topk_V2::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        input = getInputNode<float>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        input = getInputNode<int>(0);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        input = getInputNode<uint8_t>(0);

        const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
        input = DequantizeNode(input, inputIndex, ngraph::element::f32);
    }
    auto k = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    int axis = -1;  // to find largest entries for the last dimension.

    auto k_node = ngraph::opset3::Constant::create(ngraph::element::i32, {}, {k});
    const auto topk =
        std::make_shared<ngraph::opset3::TopK>(input, k_node, axis, ngraph::opset3::TopK::Mode::MAX,
                                               ngraph::opset3::TopK::SortType::SORT_VALUES);

    std::shared_ptr<ngraph::Node> outputNode1, outputNode2;

    if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        const auto& outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
        outputNode1 =
            std::make_shared<ngraph::opset3::Convert>(topk->output(0), ngraph::element::f32);
        outputNode1 = QuantizeNode(outputNode1, outputIndex, ngraph::element::u8);
    } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32)) {
        outputNode1 =
            std::make_shared<ngraph::opset3::Convert>(topk->output(0), ngraph::element::i32);
    } else if (checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        outputNode1 =
            std::make_shared<ngraph::opset3::Convert>(topk->output(0), ngraph::element::f32);
    }

    outputNode2 = std::make_shared<ngraph::opset3::Convert>(topk->output(1), ngraph::element::i32);

    auto outputIndex1 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    auto outputIndex2 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 1);
    mNgraphNodes->setOutputAtOperandIndex(outputIndex1, outputNode1);
    mNgraphNodes->setOutputAtOperandIndex(outputIndex2, outputNode2);
    const auto op = sModelInfo->getOperand(outputIndex1);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(outputIndex1, outputNode1);
        addResultNode(outputIndex2, outputNode2);
    }
    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
