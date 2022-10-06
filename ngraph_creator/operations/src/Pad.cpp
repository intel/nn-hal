#include <Pad.hpp>
#undef LOG_TAG
#define LOG_TAG "Pad"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Pad::Pad(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Pad::validate() {
    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();
    if (inputRank > 4) return false;

    // TODO: Add support for low_rank
    if (inputRank < 2) return false;

    // TODO: Add Support for all_tensors_as_inputs
    const auto& padOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);

    if (!sModelInfo->isOperandLifeTimeConst(padOperandIndex)) {
        ALOGE("%s Only Constant dimensions supported now", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ov::Node> Pad::createNode() {
    // Creating input nodes
    auto inputNode = getInputNode(0);

    const auto& paddingsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    // Fetch the 2D paddings as a 1D vector, and then split it into 2
    auto paddings_2d = sModelInfo->GetConstVecOperand<int32_t>(paddingsOperandIndex);
    auto half_size = paddings_2d.size() / 2;
    std::vector<int32_t> paddings_0(half_size);
    std::vector<int32_t> paddings_1(half_size);
    for (size_t i = 0; i < half_size; i++) {
        paddings_0[i] = paddings_2d[2 * i];
        paddings_1[i] = paddings_2d[2 * i + 1];
    }
    const auto pads_begin = createConstNode(ov::element::i32, {half_size}, paddings_0);
    const auto pads_end = createConstNode(ov::element::i32, {half_size}, paddings_1);

    auto outputNode = std::make_shared<ov::opset3::Pad>(inputNode, pads_begin, pads_end,
                                                        ov::op::PadMode::CONSTANT);
    ALOGV("outputNode Shape Size : %lu", outputNode->get_shape().size());

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
