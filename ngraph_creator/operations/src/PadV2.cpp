#include <PadV2.hpp>
#undef LOG_TAG
#define LOG_TAG "PadV2"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

PadV2::PadV2(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool PadV2::validate() {
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

std::shared_ptr<ov::Node> PadV2::createNode() {
    // Creating input nodes
    auto inputNode = getInputNode(0);
    std::shared_ptr<ov::Node> pad_value;
    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        auto pad_scalar_value = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 2);
        pad_value = createConstNode(ov::element::f32, {}, convertToVector(pad_scalar_value));
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16)) {
        auto pad_scalar_value = sModelInfo->ParseOperationInput<_Float16>(mNnapiOperationIndex, 2);
        pad_value = createConstNode(ov::element::f16, {}, convertToVector(pad_scalar_value));
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) ||
               checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED)) {
        auto pad_scalar_value = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
        pad_value = createConstNode(ov::element::i32, {}, convertToVector(pad_scalar_value));

        // scale and zeropoint of pad value has to be same as in inputNode. so inputIndex is passed
        // as second parameter to DequantizeNode
        pad_value = DequantizeNode(pad_value, inputIndex, ov::element::f32);
    }

    const auto& paddingsOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    // Fetch the 2D paddings as a 1D vector, and then split it into 2
    auto paddings_2d = sModelInfo->GetConstVecOperand<int32_t>(paddingsOperandIndex);
    auto half_size = paddings_2d.size() / 2;
    std::vector<int32_t> paddings_0(half_size);
    std::vector<int32_t> paddings_1(half_size);
    for (unsigned long i = 0; i < half_size; i++) {
        paddings_0[i] = paddings_2d[2 * i];
        paddings_1[i] = paddings_2d[2 * i + 1];
    }
    const auto pads_begin = createConstNode(ov::element::i32, {half_size}, paddings_0);
    const auto pads_end = createConstNode(ov::element::i32, {half_size}, paddings_1);

    auto outputNode = std::make_shared<ov::opset3::Pad>(inputNode, pads_begin, pads_end, pad_value,
                                                        ov::op::PadMode::CONSTANT);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
