#include <BatchToSpace.hpp>
#undef LOG_TAG
#define LOG_TAG "BatchToSpace"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

BatchToSpace::BatchToSpace(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool BatchToSpace::validate() {
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

std::shared_ptr<ov::Node> BatchToSpace::createNode() {
    int32_t layout = 0;
    bool useNchw = false;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %lu", __func__, inputsSize);

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

    const auto block_shape_node = createConstNode(ov::element::i64, {inDims.size()}, block_shape);
    const auto crop_begin = createConstNode(ov::element::i64, {shape.size()}, shape);
    const auto crop_end = createConstNode(ov::element::i64, {shape.size()}, shape);

    if (!useNchw)  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);

    std::shared_ptr<ov::Node> outputNode = std::make_shared<ov::opset3::BatchToSpace>(
        inputNode, block_shape_node, crop_begin, crop_end);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
