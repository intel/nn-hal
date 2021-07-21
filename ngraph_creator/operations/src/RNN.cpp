#include <RNN.hpp>
// Helper funciton
#include <NgraphHelper.hpp>
#define LOG_TAG "RNN"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

RNN::RNN(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool RNN::validate() {
    // check output type
    for (int i = 0; i < 2; i++) {
        if (!checkOutputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) {
            return false;
        }
    }

    // Check all input types
    for (int i = 0; i < 5; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) {
            return false;
        }
    }

    return true;
}

void RNN::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> RNN::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input, W, R, bias, initial_hidden_state;

    input = getInputNode(0);
    W = getInputNode(1);
    R = getInputNode(2);
    bias = getInputNode(3);
    initial_hidden_state = getInputNode(4);

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);

    // inputs * input_weights
    auto input_W = std::make_shared<ngraph::opset3::MatMul>(input, W, false, true);
    // state * recurrent_weights
    auto Ht_R = std::make_shared<ngraph::opset3::MatMul>(initial_hidden_state, R, false, true);
    // (state * recurrent_weights) + bias
    auto add = std::make_shared<ngraph::opset3::Add>(Ht_R, bias);
    // (inputs * input_weights) + (state * recurrent_weights) + bias
    auto i_t = std::make_shared<ngraph::opset3::Add>(input_W, add);

    auto outputNode = applyActivation(i_t, activationFn);

    for (int i = 0; i < 2; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        std::shared_ptr<ngraph::Node> outNode;
        if (i == 1) {
            outNode = outputNode;
        } else {
            outNode = createConstNode(outputNode->get_element_type(), outputNode->get_shape(),
                                      convertToVector(0));
        }

        mNgraphNodes->setOutputAtOperandIndex(outputIndex, outNode);
        ALOGD("%s Set Output index %d", __func__, outputIndex);
        const auto op = sModelInfo->getOperand(outputIndex);
        if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            addResultNode(outputIndex, outNode);
            ALOGD("%s Add result %d", __func__, outputIndex);
        }
    }

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
