#include <Depth_To_Space.hpp>
#define LOG_TAG "Depth_To_Space"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Depth_To_Space::Depth_To_Space(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Depth_To_Space::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Depth_To_Space::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;
    bool useNchw = false;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    if (inputsSize == 3) {
        auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 2);
        if (layout) useNchw = true;
    }

    input = getInputNode(0);
    auto block_size = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);

    if (!useNchw)  // No conversion needed if useNchw set
        input = transpose(NHWC_NCHW, input);

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::DepthToSpace>(
        input, ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, block_size);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
