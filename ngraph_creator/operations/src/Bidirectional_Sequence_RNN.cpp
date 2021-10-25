#include <Bidirectional_Sequence_RNN.hpp>
// Helper funciton
#include <NgraphHelper.hpp>
#undef LOG_TAG
#define LOG_TAG "Bidirectional_Sequence_RNN"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Bidirectional_Sequence_RNN::Bidirectional_Sequence_RNN(int operationIndex)
    : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Bidirectional_Sequence_RNN::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

void Bidirectional_Sequence_RNN::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> Bidirectional_Sequence_RNN::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;
    std::shared_ptr<ngraph::Node> fwWeights, fwRecurrentWeights, fwBias, fwHiddenState;
    std::shared_ptr<ngraph::Node> bwWeights, bwRecurrentWeights, bwBias, bwHiddenState;
    std::shared_ptr<ngraph::Node> auxInput, fwAuxWeights, bwAuxWeights;
    bool hasAuxInputs = false;

    input = getInputNode(0);

    fwWeights = getInputNode(1);
    fwRecurrentWeights = getInputNode(2);
    fwBias = getInputNode(3);
    fwHiddenState = getInputNode(4);

    bwWeights = getInputNode(5);
    bwRecurrentWeights = getInputNode(6);
    bwBias = getInputNode(7);
    bwHiddenState = getInputNode(8);

    auxInput = getInputNode(9);
    fwAuxWeights = getInputNode(10);
    bwAuxWeights = getInputNode(11);

    if (!isValidInputTensor(9) || !isValidInputTensor(10) || !isValidInputTensor(11)) {
        removeInputNode(9);
        removeInputNode(10);
        removeInputNode(11);
    } else {
        hasAuxInputs = true;
    }

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 12);
    auto isTimeMajor = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 13);
    auto mergeOutputs = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 14);

    const auto& inDims = getInputOperandDimensions(0);
    uint32_t maxTime;

    if (isTimeMajor) {
        maxTime = inDims[0];
    } else {
        maxTime = inDims[1];
    }

    if (!isTimeMajor) {
        input = transpose(BTS_TBS, input);
        if (hasAuxInputs) {
            auxInput = transpose(BTS_TBS, auxInput);
        }
    }

    auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(0));
    auto numSplits = maxTime;

    std::vector<ngraph::Output<ngraph::Node>> inputSplit, auxInputSplit;

    inputSplit = std::make_shared<ngraph::opset3::Split>(input, axisNode, numSplits)->outputs();

    if (hasAuxInputs) {
        auxInputSplit =
            std::make_shared<ngraph::opset3::Split>(auxInput, axisNode, numSplits)->outputs();
    }

    std::vector<std::shared_ptr<ngraph::Node>> fw_output_at_each_timestep(maxTime);
    std::vector<std::shared_ptr<ngraph::Node>> bw_output_at_each_timestep(maxTime);

    for (uint32_t i = 0; i < maxTime; i++) {
        auto dims = createConstNode(ngraph::element::i32, {0}, std::vector<int64_t>{});
        inputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(inputSplit[i], dims);

        /* ########### Forward direction ########### */
        // inputs * input_weights
        auto fw_input_W =
            std::make_shared<ngraph::opset3::MatMul>(inputSplit[i], fwWeights, false, true);
        // state * recurrent_weights
        auto fw_Ht_R = std::make_shared<ngraph::opset3::MatMul>(fwHiddenState, fwRecurrentWeights,
                                                                false, true);
        // (state * recurrent_weights) + bias
        auto fw_add = std::make_shared<ngraph::opset3::Add>(fw_Ht_R, fwBias);

        std::shared_ptr<ngraph::Node> fw_i_t;

        if (hasAuxInputs) {
            auxInputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(auxInputSplit[i], dims);
            // aux_input * aux_input_weights
            auto aux_mul = std::make_shared<ngraph::opset3::MatMul>(auxInputSplit[i], fwAuxWeights,
                                                                    false, true);
            auto fw_aux_add = std::make_shared<ngraph::opset3::Add>(aux_mul, fw_add);
            // (inputs * input_weights) + (state * recurrent_weights) + (aux_input *
            // aux_input_weights) + bias
            fw_i_t = std::make_shared<ngraph::opset3::Add>(fw_input_W, fw_aux_add);
        } else {
            // (inputs * input_weights) + (state * recurrent_weights) + bias
            fw_i_t = std::make_shared<ngraph::opset3::Add>(fw_input_W, fw_add);
        }

        auto fw_output = applyActivation(fw_i_t, activationFn);

        fwHiddenState = fw_output;
        fw_output_at_each_timestep[i] = fw_output;
    }

    for (int i = maxTime - 1; i >= 0; --i) {
        auto dims = createConstNode(ngraph::element::i32, {0}, std::vector<int64_t>{});
        inputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(inputSplit[i], dims);

        /* ########### Backward direction ########### */
        // inputs * input_weights
        auto bw_input_W =
            std::make_shared<ngraph::opset3::MatMul>(inputSplit[i], bwWeights, false, true);
        // state * recurrent_weights
        auto bw_Ht_R = std::make_shared<ngraph::opset3::MatMul>(bwHiddenState, bwRecurrentWeights,
                                                                false, true);
        // (state * recurrent_weights) + bias
        auto bw_add = std::make_shared<ngraph::opset3::Add>(bw_Ht_R, bwBias);

        std::shared_ptr<ngraph::Node> bw_i_t;

        if (hasAuxInputs) {
            auxInputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(auxInputSplit[i], dims);
            // aux_input * aux_input_weights
            auto aux_mul = std::make_shared<ngraph::opset3::MatMul>(auxInputSplit[i], bwAuxWeights,
                                                                    false, true);
            auto bw_aux_add = std::make_shared<ngraph::opset3::Add>(aux_mul, bw_add);
            // (inputs * input_weights) + (state * recurrent_weights) + (aux_input *
            // aux_input_weights) + bias
            bw_i_t = std::make_shared<ngraph::opset3::Add>(bw_input_W, bw_aux_add);
        } else {
            // (inputs * input_weights) + (state * recurrent_weights) + bias
            bw_i_t = std::make_shared<ngraph::opset3::Add>(bw_input_W, bw_add);
        }

        auto bw_output = applyActivation(bw_i_t, activationFn);

        bwHiddenState = bw_output;
        bw_output_at_each_timestep[i] = bw_output;
    }

    std::shared_ptr<ngraph::Node> fwOutputNode, bwOutputNode;
    std::vector<uint32_t> fwShape, bwShape;

    /* ########### Forward direction ########### */
    fwOutputNode = std::make_shared<ngraph::opset3::Concat>(fw_output_at_each_timestep, 0);

    auto fwOutput_batch = fwOutputNode->get_shape()[0] / maxTime;
    fwShape.push_back(maxTime);
    fwShape.push_back(fwOutput_batch);
    fwShape.push_back(fwOutputNode->get_shape()[1]);

    auto fwShapeNode =
        createConstNode(ngraph::element::i32, ngraph::Shape{input->get_shape().size()}, fwShape);

    fwOutputNode = std::make_shared<ngraph::opset3::Reshape>(fwOutputNode, fwShapeNode, false);

    /* ########### Backward direction ########### */
    bwOutputNode = std::make_shared<ngraph::opset3::Concat>(bw_output_at_each_timestep, 0);

    auto bwOutput_batch = bwOutputNode->get_shape()[0] / maxTime;
    bwShape.push_back(maxTime);
    bwShape.push_back(bwOutput_batch);
    bwShape.push_back(bwOutputNode->get_shape()[1]);

    auto bwShapeNode =
        createConstNode(ngraph::element::i32, ngraph::Shape{input->get_shape().size()}, bwShape);

    bwOutputNode = std::make_shared<ngraph::opset3::Reshape>(bwOutputNode, bwShapeNode, false);

    if (!isTimeMajor) {
        fwOutputNode = transpose(BTS_TBS, fwOutputNode);
        bwOutputNode = transpose(BTS_TBS, bwOutputNode);
    }

    if (mergeOutputs) {
        std::vector<std::shared_ptr<ngraph::Node>> concat_output;
        concat_output.push_back(fwOutputNode);
        concat_output.push_back(bwOutputNode);
        fwOutputNode = std::make_shared<ngraph::opset3::Concat>(concat_output, 2);
    }

    auto fwOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    mNgraphNodes->setOutputAtOperandIndex(fwOutputIndex, fwOutputNode);
    ALOGD("%s Set Output index %d", __func__, fwOutputIndex);
    const auto fwOp = sModelInfo->getOperand(fwOutputIndex);
    if (fwOp.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(fwOutputIndex, fwOutputNode);
        ALOGD("%s Add result %d", __func__, fwOutputIndex);
    }

    if (!mergeOutputs) {
        auto bwOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 1);
        mNgraphNodes->setOutputAtOperandIndex(bwOutputIndex, bwOutputNode);
        ALOGD("%s Set Output index %d", __func__, bwOutputIndex);
        const auto bwOp = sModelInfo->getOperand(bwOutputIndex);
        if (bwOp.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            addResultNode(bwOutputIndex, bwOutputNode);
            ALOGD("%s Add result %d", __func__, bwOutputIndex);
        }
    }

    return nullptr;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
