#include <Channel_Shuffle.hpp>
#define LOG_TAG "Channel_Shuffle"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Channel_Shuffle::Channel_Shuffle(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Channel_Shuffle::validate() {
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
    if (!checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        return false;
    }
    if (!checkInputOperandType(2, (int32_t)OperandType::INT32)) {
        return false;
    }

    // Check input rank
    const auto inputRank = getInputOperandDimensions(0).size();
    if (inputRank > 4) return false;

    return true;
}

std::shared_ptr<ngraph::Node> Channel_Shuffle::createNode() {
    // Creating input nodes
    auto inputNode = getInputNode(0);
    auto group = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);

    auto outputNode = std::make_shared<ngraph::opset3::ShuffleChannels>(inputNode, axis, group);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
