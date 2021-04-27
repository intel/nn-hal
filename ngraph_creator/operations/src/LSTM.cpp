//#define LOG_NDEBUG 0
#include <LSTM.hpp>
#define LOG_TAG "LSTM"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

#define ACTIVATION_FUNCTION_NONE 0
#define ACTIVATION_FUNCTION_RELU 1
#define ACTIVATION_FUNCTION_RELU6 3
#define ACTIVATION_FUNCTION_TANH 4
#define ACTIVATION_FUNCTION_SIGMOID 6

LSTM::LSTM(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool LSTM::validate() {
    // Check all Output types
    for (int i = 0; i <= 3; i++) {
        // Check iscratch_buffer, output state(h_t), cell state(C_t) and output(o_t) are of type
        // TENSOR_FLOAT32
        if (!checkOutputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    const auto& outputsSize = sModelInfo->getOperationOutputsSize(mNnapiOperationIndex);

    if (inputsSize != 23) {
        if (inputsSize != 27) return false;
    }

    if (outputsSize != 4) return false;

    // check 0, 18, 19 input values
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    if (!checkInputOperandType(18, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    if (!checkInputOperandType(19, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    // check input type for 2,3,4
    for (int i = 2; i <= 4; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // check input type for 6,7,8
    for (int i = 6; i <= 8; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // check input type for 13,14,15
    for (int i = 13; i <= 15; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // check input activation type
    if (!checkInputOperandType(20, (int32_t)OperandType::INT32)) {
        return false;
    }
    // check input clipping threashold for cell state and output projection layer
    for (int i = 21; i <= 22; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::FLOAT32)) return false;
    }

    if (inputsSize == 27) {
        for (int i = 23; i <= 26; i++) {
            if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        }
    }

    if (!sModelInfo->isOperationInputNull(mNnapiOperationIndex, 1) &&
        !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 5) &&
        !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 12)) {
        // CIFG diabled, check input types
        if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(5, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(12, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    if (!sModelInfo->isOperationInputNull(mNnapiOperationIndex, 9) &&
        !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 10) &&
        !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 11)) {
        // peephole enabled, check input types
        if (!checkInputOperandType(9, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(10, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(11, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    if (!sModelInfo->isOperationInputNull(mNnapiOperationIndex, 16)) {
        // projection used, check input types
        if (!checkInputOperandType(16, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> LSTM::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    bool isCIFGenabled = false, isPeepholeUsed = false, isProjectionUsed = false,
         isLayerNormUsed = false;

    // checking if CIFG enabled
    if (sModelInfo->isOperationInputNull(mNnapiOperationIndex, 1) &&
        sModelInfo->isOperationInputNull(mNnapiOperationIndex, 5) &&
        sModelInfo->isOperationInputNull(mNnapiOperationIndex, 12)) {
        isCIFGenabled = true;
    }

    // checking if peephole enabled
    const auto& c2iDimensions = getInputOperandDimensions(9);
    const auto& c2fDimensions = getInputOperandDimensions(10);
    const auto& c2oDimensions = getInputOperandDimensions(11);

    if (!sModelInfo->isOperationInputNull(mNnapiOperationIndex, 9) &&
        !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 10) &&
        !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 11)) {
        isPeepholeUsed = true;
    }

    if (!c2iDimensions.empty() && c2iDimensions[0] != 0)
        isPeepholeUsed = true;
    else
        isPeepholeUsed = false;

    // checking if projection enabled
    const auto& projWeightsDims = getInputOperandDimensions(16);
    if (!sModelInfo->isOperationInputNull(mNnapiOperationIndex, 16)) {
        isProjectionUsed = true;
    }

    if (!projWeightsDims.empty() && projWeightsDims[0] != 0)
        isProjectionUsed = true;
    else
        isProjectionUsed = false;

    if (inputsSize == 27) {
        // checking if layer normalization enabled
        if (!sModelInfo->isOperationInputNull(mNnapiOperationIndex, 23) &&
            !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 24) &&
            !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 25) &&
            !sModelInfo->isOperationInputNull(mNnapiOperationIndex, 26)) {
            isLayerNormUsed = true;
        }
    }

    // Create input, initial output state, initail cell state nodes
    auto inputNode = getInputNode<float>(0);
    auto initial_hidden_state = getInputNode<float>(18);  // h_{t-1}
    auto initial_cell_state = getInputNode<float>(19);    // C_{t-1}

    const auto& initial_hidden_state_dims = getInputOperandDimensions(18);
    const auto& initial_cell_state_dims = getInputOperandDimensions(19);
    auto hidden_size = initial_cell_state_dims[1];
    auto output_size = initial_hidden_state_dims[1];

    // Create input weight nodes W_{xi}, W_{xf}, W_{xc}, W_{xo}
    auto input2input_weights = getInputNode<float>(1);  // optional, for CIFG no value
    auto input2forget_weights = getInputNode<float>(2);
    auto input2cell_weights = getInputNode<float>(3);
    auto input2output_weights = getInputNode<float>(4);

    // Create reccurence weight nodes W_{hi}, W_{hf}, W_{hc}, W_{ho}
    auto recurrent2input_weights = getInputNode<float>(
        5);  // optional, for CIFG no value and also changes output size if projection is defined
    auto recurrent2forget_weights = getInputNode<float>(6);
    auto recurrent2cell_weights = getInputNode<float>(7);
    auto recurrent2output_weights = getInputNode<float>(8);

    // Create bias nodes b_i, b_f, b_c, b_o
    auto input_gate_bias = getInputNode<float>(12);  // optional, for CIFG no value
    auto forget_gate_bias = getInputNode<float>(13);
    auto cell_bias = getInputNode<float>(14);
    auto output_gate_bias = getInputNode<float>(15);

    // Create weight, reccurence and bias tensors W, R, B
    auto W = make_shared<ngraph::opset3::Concat>(
        ngraph::NodeVector{
            transpose(NC_CN, input2input_weights), transpose(NC_CN, input2forget_weights),
            transpose(NC_CN, input2cell_weights), transpose(NC_CN, input2output_weights)},
        1);
    auto R = make_shared<ngraph::opset3::Concat>(
        ngraph::NodeVector{
            transpose(NC_CN, recurrent2input_weights), transpose(NC_CN, recurrent2forget_weights),
            transpose(NC_CN, recurrent2cell_weights), transpose(NC_CN, recurrent2output_weights)},
        1);
    auto B = make_shared<ngraph::opset3::Concat>(
        ngraph::NodeVector{input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias}, 0);

    std::shared_ptr<ngraph::Node> cell2input_weights, cell2forget_weights, cell2output_weights;

    if (isPeepholeUsed) {
        // optional peephole parameters W_{ci}, W_{cf}, W_{co}
        cell2input_weights = getInputNode<float>(9);
        if (!c2fDimensions.empty()) {
            if (c2fDimensions[0] != 0) {
                cell2forget_weights = getInputNode<float>(10);
            } else {
                cell2forget_weights = std::make_shared<ngraph::opset3::Constant>(
                    inputNode->get_element_type(), ngraph::Shape{hidden_size},
                    std::vector<float>{0.f});
            }
        } else {
            cell2forget_weights = std::make_shared<ngraph::opset3::Constant>(
                inputNode->get_element_type(), ngraph::Shape{hidden_size}, std::vector<float>{0.f});
        }
        if (!c2oDimensions.empty()) {
            if (c2oDimensions[0] != 0) {
                cell2output_weights = getInputNode<float>(11);
            } else {
                cell2output_weights = std::make_shared<ngraph::opset3::Constant>(
                    inputNode->get_element_type(), ngraph::Shape{hidden_size},
                    std::vector<float>{0.f});
            }
        } else {
            cell2output_weights = std::make_shared<ngraph::opset3::Constant>(
                inputNode->get_element_type(), ngraph::Shape{hidden_size}, std::vector<float>{0.f});
        }

    } else {
        // Create default peephole
        cell2input_weights = std::make_shared<ngraph::opset3::Constant>(
            inputNode->get_element_type(), ngraph::Shape{hidden_size}, std::vector<float>{0.f});
        cell2forget_weights = std::make_shared<ngraph::opset3::Constant>(
            inputNode->get_element_type(), ngraph::Shape{hidden_size}, std::vector<float>{0.f});
        cell2output_weights = std::make_shared<ngraph::opset3::Constant>(
            inputNode->get_element_type(), ngraph::Shape{hidden_size}, std::vector<float>{0.f});
    }

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 20);

    auto cell_state_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 21);

    std::shared_ptr<ngraph::Node> scratchBuffer;

    auto axisNode = ngraph::op::Constant::create(ngraph::element::u64, ngraph::Shape{}, {1});

    // Xt*(W^T) -- for [iofc] gates.
    auto Xt_W = matMul(inputNode, W, false, false);

    // Ht-1*(R^T)  -- for [iofc] gates.
    auto Ht_R = matMul(initial_hidden_state, R, false, false);

    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
    auto gates = add(Xt_W, add(Ht_R, B));

    auto split_gates = std::make_shared<ngraph::opset3::Split>(gates, axisNode, 4);

    auto i_t = split_gates->output(0);
    auto f_t = split_gates->output(1);
    auto c_t = split_gates->output(2);
    auto o_t = split_gates->output(3);

    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    i_t = applyActivation(
        clip(add(i_t, mul(cell2input_weights, initial_cell_state)), cell_state_clipping), 6);

    if (isCIFGenabled) {
        // Couple input with forget gate: 1 - i_t
        f_t =
            sub(ngraph::op::Constant::create(i_t.get_element_type(), i_t.get_shape(),
                                             std::vector<float>(shape_size(i_t.get_shape()), 1.f)),
                i_t);
        scratchBuffer = std::make_shared<ngraph::opset3::Constant>(
            inputNode->get_element_type(), ngraph::Shape{1, 3 * hidden_size},
            std::vector<float>{0.f});
    } else {
        // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        f_t = applyActivation(
            clip(add(f_t, mul(cell2forget_weights, initial_cell_state)), cell_state_clipping), 6);
        scratchBuffer = std::make_shared<ngraph::opset3::Constant>(
            inputNode->get_element_type(), ngraph::Shape{1, 4 * hidden_size},
            std::vector<float>{0.f});
    }

    // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, initial_cell_state),
                 mul(i_t, applyActivation(clip(c_t, cell_state_clipping), activationFn)));  // C_t

    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = applyActivation(clip(add(o_t, mul(cell2output_weights, C)), cell_state_clipping),
                          6);  // o_t

    std::shared_ptr<ngraph::Node> H;
    if (isProjectionUsed) {
        auto projection_weights = getInputNode<float>(16);
        std::shared_ptr<ngraph::Node> projection_bias;
        const auto& projBiasDims = getInputOperandDimensions(17);
        float p_clip;
        if (!projBiasDims.empty()) {
            if (projBiasDims[0] != 0) {
                projection_bias = getInputNode<float>(17);
            } else {
                projection_bias = std::make_shared<ngraph::opset3::Constant>(
                    inputNode->get_element_type(), ngraph::Shape{output_size},
                    std::vector<float>{0.f});
            }
        } else {
            projection_bias = std::make_shared<ngraph::opset3::Constant>(
                inputNode->get_element_type(), ngraph::Shape{output_size}, std::vector<float>{0.f});
        }
        p_clip = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 22);
        auto projWeightsProduct = matMul(
            projection_weights,
            mul(o_t, applyActivation(clip(C, cell_state_clipping), activationFn)), false, true);
        // clip(W_{proj}(o_t odot g(C_t))+b_{proj}, t_{proj})
        H = clip(add(transpose(NC_CN, projWeightsProduct), projection_bias), p_clip);  // h_t
    } else {
        // ot (.) h(Ct)
        H = mul(o_t, applyActivation(clip(C, cell_state_clipping), activationFn));  // h_t
    }

    std::vector<std::shared_ptr<ngraph::Node>> LstmOutputs(4, nullptr);
    LstmOutputs[0] = scratchBuffer;
    LstmOutputs[1] = H;
    LstmOutputs[2] = C;
    LstmOutputs[3] = o_t.get_node_shared_ptr();

    // eleminating scratchBuffer, otherwise its crashing loadNetwork()
    for (int i = 0; i < 4; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        if (i == 0)
            mNgraphNodes->setInvalidNode(outputIndex);
        else {
            mNgraphNodes->setOutputAtOperandIndex(outputIndex, LstmOutputs[i]);
            const auto op = sModelInfo->getOperand(outputIndex);
            if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
                addResultNode(outputIndex, LstmOutputs[i]);
            }
        }
    }
    return scratchBuffer;
}

std::shared_ptr<ngraph::Node> LSTM::add(const ngraph::Output<ngraph::Node>& lhs,
                                        const ngraph::Output<ngraph::Node>& rhs) {
    return {make_shared<ngraph::opset3::Add>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> LSTM::sub(const ngraph::Output<ngraph::Node>& lhs,
                                        const ngraph::Output<ngraph::Node>& rhs) {
    return {make_shared<ngraph::opset3::Add>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> LSTM::mul(const ngraph::Output<ngraph::Node>& lhs,
                                        const ngraph::Output<ngraph::Node>& rhs) {
    return {make_shared<ngraph::opset3::Multiply>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> LSTM::matMul(const ngraph::Output<ngraph::Node>& lhs,
                                           const ngraph::Output<ngraph::Node>& rhs,
                                           bool transpose_lhs, bool transpose_rhs) {
    return {make_shared<ngraph::opset3::MatMul>(lhs, rhs, transpose_lhs, transpose_rhs)};
}

std::shared_ptr<ngraph::Node> LSTM::clip(const ngraph::Output<ngraph::Node>& data,
                                         float m_clip) const {
    if (m_clip == 0.f) {
        return data.get_node_shared_ptr();
    }
    return make_shared<ngraph::opset3::Clamp>(data, -m_clip, m_clip);
}
std::shared_ptr<ngraph::Node> LSTM::applyActivation(const std::shared_ptr<ngraph::Node>& arg,
                                                    int activationFn) const {
    switch (activationFn) {
        case ACTIVATION_FUNCTION_RELU:
            return std::make_shared<ngraph::opset3::Relu>(arg);
            break;
        case ACTIVATION_FUNCTION_RELU6:
            return std::make_shared<ngraph::opset3::Clamp>(arg, 0, 6);
            break;
        case ACTIVATION_FUNCTION_TANH:
            return std::make_shared<ngraph::opset3::Tanh>(arg);
            break;
        case ACTIVATION_FUNCTION_SIGMOID:
            return std::make_shared<ngraph::opset3::Sigmoid>(arg);
            break;
        default:
            return std::make_shared<ngraph::opset3::Sigmoid>(arg);
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
