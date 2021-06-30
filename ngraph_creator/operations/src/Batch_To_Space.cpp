//#define LOG_NDEBUG 0
#include <Batch_To_Space.hpp>
#define LOG_TAG "Batch_To_Space"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Batch_To_Space::Batch_To_Space(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Batch_To_Space::validate() {
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    if (inputsSize == 3) {
        if (!checkInputOperandType(2, (int32_t)OperandType::BOOL)) {
            return false;
        }
    }

    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();

    if (inputRank != 4) return false;

    const auto& block_shape_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    // TODO: Add Support for all_tensors_as_inputs
    if (!sModelInfo->isOperandLifeTimeConst(block_shape_OperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Batch_To_Space::createNode() {
    int32_t layout = 0;
    bool useNchw = false;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %d", __func__, inputsSize);

    auto inputNode = getInputNode(0);
    auto& inDims = getInputOperandDimensions(0);
    const auto& block_shape_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    auto block_shape = sModelInfo->GetConstVecOperand<int32_t>(block_shape_OperandIndex);

    // Compensation for the shape to be same as the size of data input shape
    block_shape.insert(block_shape.begin(), 1);
    block_shape.insert(block_shape.begin(), 1);

    if (inputsSize == 3) layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);
    if (layout) useNchw = true;

    std::vector<uint32_t> shape(inDims.size(), 0);

    const auto block_shape_node =
        createConstNode(ngraph::element::i64, {inDims.size()}, block_shape);
    const auto crop_begin = createConstNode(ngraph::element::i64, {shape.size()}, shape);
    const auto crop_end = createConstNode(ngraph::element::i64, {shape.size()}, shape);

    if (!useNchw)  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::BatchToSpace>(
        inputNode, block_shape_node, crop_begin, crop_end);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
