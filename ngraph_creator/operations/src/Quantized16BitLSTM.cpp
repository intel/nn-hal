#include <Quantized16BitLSTM.hpp>
#undef LOG_TAG
#define LOG_TAG "Quantized16BitLSTM"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

#define ACTIVATION_FUNCTION_NONE 0
#define ACTIVATION_FUNCTION_RELU 1
#define ACTIVATION_FUNCTION_RELU6 3
#define ACTIVATION_FUNCTION_TANH 4
#define ACTIVATION_FUNCTION_SIGMOID 6

Quantized16BitLSTM::Quantized16BitLSTM(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Quantized16BitLSTM::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

void Quantized16BitLSTM::connectOperationToGraph() { createNode(); }

// TODO: Fix VTS Generated test cases; output mismatch error
std::shared_ptr<ngraph::Node> Quantized16BitLSTM::createNode() {
    // Creating input nodes
    auto inputNode = getInputNode(0);

    auto input2input_weights = getInputNode(1);
    auto input2forget_weights = getInputNode(2);
    auto input2cell_weights = getInputNode(3);
    auto input2output_weights = getInputNode(4);

    auto recurrent2input_weights = getInputNode(5);
    auto recurrent2forget_weights = getInputNode(6);
    auto recurrent2cell_weights = getInputNode(7);
    auto recurrent2output_weights = getInputNode(8);

    auto input_gate_bias = getInputNode(9);
    auto forget_gate_bias = getInputNode(10);
    auto cell_gate_bias = getInputNode(11);
    auto output_gate_bias = getInputNode(12);

    auto input_gate_bias_index = sModelInfo->getOperationInput(mNnapiOperationIndex, 9);
    auto forget_gate_bias_index = sModelInfo->getOperationInput(mNnapiOperationIndex, 10);
    auto cell_gate_bias_index = sModelInfo->getOperationInput(mNnapiOperationIndex, 11);
    auto output_gate_bias_index = sModelInfo->getOperationInput(mNnapiOperationIndex, 12);

    input_gate_bias = DequantizeNode(input_gate_bias, input_gate_bias_index, ngraph::element::f32);
    forget_gate_bias =
        DequantizeNode(forget_gate_bias, forget_gate_bias_index, ngraph::element::f32);
    cell_gate_bias = DequantizeNode(cell_gate_bias, cell_gate_bias_index, ngraph::element::f32);
    output_gate_bias =
        DequantizeNode(output_gate_bias, output_gate_bias_index, ngraph::element::f32);

    auto initial_cell_state = getInputNode(13);
    auto initial_hidden_state = getInputNode(14);

    // i_t = W_{xi}x_t+W_{hi}h_{t-1} + b_i
    auto i_t = add(add(matMul(inputNode, input2input_weights, false, true),
                       matMul(initial_hidden_state, recurrent2input_weights, false, true)),
                   input_gate_bias);
    // f_t = W_{xf}x_t+W_{hf}h_{t-1} + b_f
    auto f_t = add(add(matMul(inputNode, input2forget_weights, false, true),
                       matMul(initial_hidden_state, recurrent2forget_weights, false, true)),
                   forget_gate_bias);
    // c_t = W_{xc}x_t+W_{hc}h_{t-1} + b_c
    auto c_t = add(add(matMul(inputNode, input2cell_weights, false, true),
                       matMul(initial_hidden_state, recurrent2cell_weights, false, true)),
                   cell_gate_bias);
    // o_t = W_{xo}x_t+W_{ho}h_{t-1} + b_o
    auto o_t = add(add(matMul(inputNode, input2output_weights, false, true),
                       matMul(initial_hidden_state, recurrent2output_weights, false, true)),
                   output_gate_bias);

    // sigma(W_{xi}x_t + W_{hi}h_{t-1} + + b_i)
    i_t = applyActivation(i_t, ACTIVATION_FUNCTION_SIGMOID);
    // sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)
    f_t = applyActivation(f_t, ACTIVATION_FUNCTION_SIGMOID);
    // g(W_{xc}x_t+W_{hc}h_{t-1}+b_c)
    c_t = applyActivation(c_t, ACTIVATION_FUNCTION_TANH);
    // sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o)
    o_t = applyActivation(o_t, ACTIVATION_FUNCTION_SIGMOID);

    // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, initial_cell_state), mul(i_t, c_t));
    // o_t odot g(C_t)
    auto H = mul(o_t, applyActivation(C, ACTIVATION_FUNCTION_TANH));

    std::vector<std::shared_ptr<ngraph::Node>> LstmOutputs(2, nullptr);
    LstmOutputs[0] = C;
    LstmOutputs[1] = H;

    for (int i = 0; i < 2; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        const auto op = sModelInfo->getOperand(outputIndex);
        std::shared_ptr<ngraph::Node> outnode;
        if (op.type == OperandType::TENSOR_QUANT8_ASYMM) {
            outnode = QuantizeNode(LstmOutputs[i], outputIndex, ngraph::element::u8);
        } else if (op.type == OperandType::TENSOR_QUANT16_SYMM) {
            outnode = QuantizeNode(LstmOutputs[i], outputIndex, ngraph::element::i16);
        }
        mNgraphNodes->setOutputAtOperandIndex(outputIndex, outnode);
        if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(outputIndex, outnode);
        }
    }

    return nullptr;
}

std::shared_ptr<ngraph::Node> Quantized16BitLSTM::applyActivation(
    const std::shared_ptr<ngraph::Node>& arg, int activationFn) const {
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
            return std::make_shared<ngraph::opset3::Tanh>(arg);
    }
}

std::shared_ptr<ngraph::Node> Quantized16BitLSTM::add(const ngraph::Output<ngraph::Node>& lhs,
                                                      const ngraph::Output<ngraph::Node>& rhs) {
    return {std::make_shared<ngraph::opset3::Add>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> Quantized16BitLSTM::sub(const ngraph::Output<ngraph::Node>& lhs,
                                                      const ngraph::Output<ngraph::Node>& rhs) {
    return {
        std::make_shared<ngraph::opset3::Subtract>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> Quantized16BitLSTM::mul(const ngraph::Output<ngraph::Node>& lhs,
                                                      const ngraph::Output<ngraph::Node>& rhs) {
    return {
        std::make_shared<ngraph::opset3::Multiply>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> Quantized16BitLSTM::matMul(const ngraph::Output<ngraph::Node>& lhs,
                                                         const ngraph::Output<ngraph::Node>& rhs,
                                                         bool transpose_lhs, bool transpose_rhs) {
    return {std::make_shared<ngraph::opset3::MatMul>(lhs, rhs, transpose_lhs, transpose_rhs)};
}

std::shared_ptr<ngraph::Node> Quantized16BitLSTM::clip(const ngraph::Output<ngraph::Node>& data,
                                                       float m_clip) const {
    if (m_clip == 0.f) {
        return data.get_node_shared_ptr();
    }
    return std::make_shared<ngraph::opset3::Clamp>(data, -m_clip, m_clip);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
