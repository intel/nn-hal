#include <Unidirectional_Sequence_RNN.hpp>
// Helper funciton
#include <NgraphHelper.hpp>
#define LOG_TAG "Unidirectional_Sequence_RNN"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Unidirectional_Sequence_RNN::Unidirectional_Sequence_RNN(int operationIndex)
    : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Unidirectional_Sequence_RNN::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Unidirectional_Sequence_RNN::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> inputNode, W, R, bias, initial_hidden_state, outputNode;

    inputNode = getInputNode(0);
    W = getInputNode(1);
    R = getInputNode(2);
    bias = getInputNode(3);
    initial_hidden_state = getInputNode(4);

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
    auto isTimeMajor = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

    if (!isTimeMajor) {
        inputNode = transpose(BTS_TBS, inputNode);
    }

    const auto& inDims = getInputOperandDimensions(0);
    uint32_t maxTime;

    if (isTimeMajor) {
        maxTime = inDims[0];
    } else {
        maxTime = inDims[1];
    }

    auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(0));
    auto numSplits = maxTime;

    auto inputSplit =
        std::make_shared<ngraph::opset3::Split>(inputNode, axisNode, numSplits)->outputs();

    std::vector<std::shared_ptr<ngraph::Node>> output_at_each_timestep;

    for (int i = 0; i < maxTime; i++) {
        auto dims = createConstNode(ngraph::element::i32, {0}, std::vector<int64_t>{});
        inputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(inputSplit[i], dims);
        // inputs * input_weights
        auto input_W = std::make_shared<ngraph::opset3::MatMul>(inputSplit[i], W, false, true);
        // state * recurrent_weights
        auto Ht_R = std::make_shared<ngraph::opset3::MatMul>(initial_hidden_state, R, false, true);
        // (state * recurrent_weights) + bias
        auto add = std::make_shared<ngraph::opset3::Add>(Ht_R, bias);
        // (inputs * input_weights) + (state * recurrent_weights) + bias
        auto i_t = std::make_shared<ngraph::opset3::Add>(input_W, add);
        // activation((inputs * input_weights) + (state * recurrent_weights) + bias)
        auto outNode = applyActivation(i_t, activationFn);

        initial_hidden_state = outNode;
        output_at_each_timestep.push_back(outNode);
    }

    outputNode = std::make_shared<ngraph::opset3::Concat>(output_at_each_timestep, 0);
    std::vector<uint32_t> shape;
    auto output_batch = outputNode->get_shape()[0] / maxTime;
    shape.push_back(maxTime);
    shape.push_back(output_batch);
    shape.push_back(outputNode->get_shape()[1]);

    auto shapeNode =
        createConstNode(ngraph::element::i32, ngraph::Shape{inputNode->get_shape().size()}, shape);
    outputNode = std::make_shared<ngraph::opset3::Reshape>(outputNode, shapeNode, false);
    if (!isTimeMajor) {
        outputNode = transpose(BTS_TBS, outputNode);
    }

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
