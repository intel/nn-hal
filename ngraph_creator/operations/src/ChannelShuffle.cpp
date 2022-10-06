#include <ChannelShuffle.hpp>
#undef LOG_TAG
#define LOG_TAG "ChannelShuffle"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

ChannelShuffle::ChannelShuffle(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool ChannelShuffle::validate() {
    // Check input rank
    const int64_t inputRank = getInputOperandDimensions(0).size();
    if (inputRank > 4 || inputRank <= 0) {
        ALOGE("%s Invalid input dimensions size!", __func__);
        return false;
    }

    // Check axis range
    int64_t axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);
    if (!(axis >= -inputRank && axis < inputRank)) {
        ALOGE("%s Axis %ld not in the range [-inputRank, inputRank)", __func__, axis);
        return false;
    }

    return true;
}

std::shared_ptr<ov::Node> ChannelShuffle::createNode() {
    // Creating input nodes
    auto inputNode = getInputNode(0);
    int64_t group = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);
    int64_t axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 2);

    auto inputRank = getInputOperandDimensions(0).size();
    axis = (axis >= 0) ? axis : (axis + inputRank);

    // Convert the inputNode to 4D
    std::shared_ptr<ov::Node> squeezeAxes;
    if (inputRank < 4) {
        switch (inputRank) {
            case 1:
                squeezeAxes = std::make_shared<ov::opset3::Constant>(ov::element::i64, ov::Shape{3},
                                                                     std::vector<int64_t>{1, 2, 3});
                break;
            case 2:
                squeezeAxes = std::make_shared<ov::opset3::Constant>(ov::element::i64, ov::Shape{2},
                                                                     std::vector<int64_t>{2, 3});
                break;
            case 3:
                squeezeAxes = std::make_shared<ov::opset3::Constant>(ov::element::i64, ov::Shape{1},
                                                                     std::vector<int64_t>{3});
                break;
            default:
                break;
        }

        inputNode = std::make_shared<ov::opset3::Unsqueeze>(inputNode, squeezeAxes);
    }

    std::shared_ptr<ov::Node> outputNode =
        std::make_shared<ov::opset3::ShuffleChannels>(inputNode, axis, group);

    // Using squeeze to convert the shape of outputNode to shape of inputNode before unsqueeze
    if (inputRank < 4) {
        outputNode = std::make_shared<ov::opset3::Squeeze>(outputNode, squeezeAxes);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
