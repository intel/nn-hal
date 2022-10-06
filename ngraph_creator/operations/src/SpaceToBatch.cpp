#include <SpaceToBatch.hpp>
#undef LOG_TAG
#define LOG_TAG "SpaceToBatch"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

SpaceToBatch::SpaceToBatch(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool SpaceToBatch::validate() {
    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();

    if (inputRank != 4) return false;

    auto& block_shape_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    // TODO: Add Support for all_tensors_as_inputs
    if (!sModelInfo->isOperandLifeTimeConst(block_shape_OperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    auto pad_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    // TODO: Add Support for all_tensors_as_inputs
    if (!sModelInfo->isOperandLifeTimeConst(pad_OperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ov::Node> SpaceToBatch::createNode() {
    int32_t layout = 0;
    bool useNchw = false;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    auto inputNode = getInputNode(0);

    auto& inDims = getInputOperandDimensions(0);

    const auto& block_shape_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    auto block_shape = sModelInfo->GetConstVecOperand<int32_t>(block_shape_OperandIndex);

    // Compensation for the shape to be same as the size of data input shape
    block_shape.insert(block_shape.begin(), 1);
    block_shape.insert(block_shape.begin(), 1);

    const auto& pad_OperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    // Fetch the 2D pad as a 1D vector, and then split it into 2
    auto pad_2d = sModelInfo->GetConstVecOperand<int32_t>(pad_OperandIndex);
    auto half_size = pad_2d.size() / 2;
    std::vector<int32_t> pad_0(half_size);
    std::vector<int32_t> pad_1(half_size);
    for (size_t i = 0; i < half_size; i++) {
        pad_0[i] = pad_2d[2 * i];
        pad_1[i] = pad_2d[2 * i + 1];
    }

    // Compensation for the shape to be same as the size of data input shape
    pad_0.insert(pad_0.begin(), 0);
    pad_0.insert(pad_0.begin(), 0);

    // Compensation for the shape to be same as the size of data input shape
    pad_1.insert(pad_1.begin(), 0);
    pad_1.insert(pad_1.begin(), 0);

    const auto block_shape_node = createConstNode(ov::element::i64, {inDims.size()}, block_shape);
    const auto pad_begin = createConstNode(ov::element::i64, {inDims.size()}, pad_0);
    const auto pad_end = createConstNode(ov::element::i64, {inDims.size()}, pad_1);

    if (inputsSize == 4) layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 3);
    if (layout) useNchw = true;

    if (!useNchw)  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);

    std::shared_ptr<ov::Node> outputNode =
        std::make_shared<ov::opset3::SpaceToBatch>(inputNode, block_shape_node, pad_begin, pad_end);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
