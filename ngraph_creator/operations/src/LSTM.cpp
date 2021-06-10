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

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 1) &&
        !sModelInfo->isOmittedInput(mNnapiOperationIndex, 5) &&
        !sModelInfo->isOmittedInput(mNnapiOperationIndex, 12)) {
        // CIFG diabled, check input types
        if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(5, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(12, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 9) &&
        !sModelInfo->isOmittedInput(mNnapiOperationIndex, 10) &&
        !sModelInfo->isOmittedInput(mNnapiOperationIndex, 11)) {
        // peephole enabled, check input types
        if (!checkInputOperandType(9, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(10, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(11, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    if (!sModelInfo->isOmittedInput(mNnapiOperationIndex, 16)) {
        // projection used, check input types
        if (!checkInputOperandType(16, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

void LSTM::connectOperationToGraph() { createNode(); }

std::shared_ptr<ngraph::Node> LSTM::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    bool isCIFGenabled = false, isPeepholeUsed = false, isProjectionUsed = false,
         isLayerNormUsed = false, isCifgDimsEmpty = true;

    // checking if CIFG enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 1) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 5) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 12)) {
        isCIFGenabled = true;
    } else {
        if (isValidInputTensor(1) && isValidInputTensor(5) && isValidInputTensor(12))
            isCIFGenabled = false;
        else
            isCIFGenabled = true;
    }

    // checking if peephole enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 9) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 10) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 11)) {
        isPeepholeUsed = false;
    } else {
        if (!isCIFGenabled && !isValidInputTensor(9) && isValidInputTensor(10) &&
            isValidInputTensor(11)) {
            isCIFGenabled = true;
            isCifgDimsEmpty = false;
        }
        if (isCIFGenabled) {
            if (isValidInputTensor(10) && isValidInputTensor(11))
                isPeepholeUsed = true;
            else
                isPeepholeUsed = false;
        } else {
            if (isValidInputTensor(9) && isValidInputTensor(10) && isValidInputTensor(11))
                isPeepholeUsed = true;
            else
                isPeepholeUsed = false;
        }
    }

    // checking if projection enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 16)) {
        isProjectionUsed = false;
    } else {
        if (isValidInputTensor(16))
            isProjectionUsed = true;
        else
            isProjectionUsed = false;
    }

    if (inputsSize == 27) {
        // checking if layer normalization enabled
        if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 23) &&
            sModelInfo->isOmittedInput(mNnapiOperationIndex, 24) &&
            sModelInfo->isOmittedInput(mNnapiOperationIndex, 25) &&
            sModelInfo->isOmittedInput(mNnapiOperationIndex, 26)) {
            isLayerNormUsed = false;
        } else {
            if (isCIFGenabled) {
                if (isValidInputTensor(24) && isValidInputTensor(25) && isValidInputTensor(26))
                    isLayerNormUsed = true;
                else
                    isLayerNormUsed = false;

            } else {
                if (isValidInputTensor(23) && isValidInputTensor(24) && isValidInputTensor(25) &&
                    isValidInputTensor(26))
                    isLayerNormUsed = true;
                else
                    isLayerNormUsed = false;
            }
        }
    }

    std::shared_ptr<ngraph::Node> inputNode, input2input_weights, input2forget_weights,
        input2cell_weights, input2output_weights, recurrent2input_weights, recurrent2forget_weights,
        recurrent2cell_weights, recurrent2output_weights, cell2input_weights, cell2forget_weights,
        cell2output_weights, input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias,
        projection_weights, projection_bias, initial_hidden_state, initial_cell_state;
    uint32_t activationFn;
    float cell_state_clipping, proj_clipping;

    const auto& inputNode_dims = getInputOperandDimensions(0);
    const auto& initial_hidden_state_dims = getInputOperandDimensions(18);
    const auto& initial_cell_state_dims = getInputOperandDimensions(19);

    auto batch_size = inputNode_dims[0];
    auto input_size = inputNode_dims[1];
    auto num_units = initial_cell_state_dims[1];
    auto output_size = initial_hidden_state_dims[1];

    // Creating input nodes
    inputNode = getInputNode(0);
    const auto& elementType = inputNode->get_element_type();

    // W_{xi}, W_{xf}, W_{xc}, W_{xo}
    if (isCIFGenabled) {
        if (!isCifgDimsEmpty) removeInputNode(1);
    } else {
        input2input_weights = getInputNode(1);
    }
    input2forget_weights = getInputNode(2);
    input2cell_weights = getInputNode(3);
    input2output_weights = getInputNode(4);

    // W_{hi}, W_{hf}, W_{hc}, W_{ho}
    if (isCIFGenabled) {
        if (!isCifgDimsEmpty) removeInputNode(5);
    } else {
        recurrent2input_weights = getInputNode(5);
    }
    recurrent2forget_weights = getInputNode(6);
    recurrent2cell_weights = getInputNode(7);
    recurrent2output_weights = getInputNode(8);

    // W_{ci}, W_{cf}, W_{co}
    if (isPeepholeUsed) {
        if (isCIFGenabled)
            cell2input_weights =
                createConstNode(elementType, ngraph::Shape{num_units}, convertToVector(0));
        else
            cell2input_weights = getInputNode(9);
        cell2forget_weights = getInputNode(10);
        cell2output_weights = getInputNode(11);
    } else {
        cell2input_weights =
            createConstNode(elementType, ngraph::Shape{num_units}, convertToVector(0));
        cell2forget_weights =
            createConstNode(elementType, ngraph::Shape{num_units}, convertToVector(0));
        cell2output_weights =
            createConstNode(elementType, ngraph::Shape{num_units}, convertToVector(0));
    }

    // b_i, b_f, b_c, b_o
    if (isCIFGenabled) {
        if (!isCifgDimsEmpty) removeInputNode(12);
    } else {
        input_gate_bias = getInputNode(12);
    }
    forget_gate_bias = getInputNode(13);
    cell_bias = getInputNode(14);
    output_gate_bias = getInputNode(15);

    // W_{proj}, b_{proj}
    if (isProjectionUsed) {
        projection_weights = getInputNode(16);
        if (isValidInputTensor(17))
            projection_bias = getInputNode(17);
        else
            projection_bias =
                createConstNode(elementType, ngraph::Shape{output_size}, convertToVector(0));
    }

    initial_hidden_state = getInputNode(18);  // h_{t-1}
    initial_cell_state = getInputNode(19);    // C_{t-1}

    activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 20);
    cell_state_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 21);

    if (isProjectionUsed)
        proj_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 22);

    std::shared_ptr<ngraph::Node> i_t, f_t, c_t, o_t;
    std::shared_ptr<ngraph::Node> scratchBuffer;

    std::shared_ptr<ngraph::Node> input_layer_norm_weights, forget_layer_norm_weights,
        cell_layer_norm_weights, output_layer_norm_weights;

    std::shared_ptr<ngraph::Node> i_t_Mean, i_t_Variance, f_t_Mean, f_t_Variance, c_t_Mean,
        c_t_Variance, o_t_Mean, o_t_Variance;

    std::shared_ptr<ngraph::Node> reduceAxes;

    if (isLayerNormUsed) {
        if (!isCIFGenabled) input_layer_norm_weights = getInputNode(23);
        forget_layer_norm_weights = getInputNode(24);
        cell_layer_norm_weights = getInputNode(25);
        output_layer_norm_weights = getInputNode(26);
        reduceAxes = createConstNode(ngraph::element::i32, {}, convertToVector(0));
    }

    // i_t = W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}
    if (!isCIFGenabled)
        i_t = add(add(matMul(inputNode, input2input_weights, false, true),
                      matMul(initial_hidden_state, recurrent2input_weights, false, true)),
                  mul(cell2input_weights, initial_cell_state));
    // f_t = W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}
    f_t = add(add(matMul(inputNode, input2forget_weights, false, true),
                  matMul(initial_hidden_state, recurrent2forget_weights, false, true)),
              mul(cell2forget_weights, initial_cell_state));
    // c_t = W_{xc}x_t+W_{hc}h_{t-1}
    c_t = add(matMul(inputNode, input2cell_weights, false, true),
              matMul(initial_hidden_state, recurrent2cell_weights, false, true));
    // o_t = W_{xo}x_t+W_{ho}h_{t-1}
    o_t = add(matMul(inputNode, input2output_weights, false, true),
              matMul(initial_hidden_state, recurrent2output_weights, false, true));

    /* ################# Update Forget Gate ################# */
    if (isLayerNormUsed) {
        f_t_Mean = calculatemean(f_t, reduceAxes, false);
        f_t_Variance = calculateVariance(f_t, f_t_Mean, reduceAxes, elementType);
        f_t = LayerNorm(f_t, f_t_Mean, f_t_Variance, forget_layer_norm_weights, forget_gate_bias);
    } else {
        // W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}C_{t-1} + b_f
        f_t = add(f_t, forget_gate_bias);
    }
    // sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}C_{t-1} + b_f)
    f_t = applyActivation(f_t, ACTIVATION_FUNCTION_SIGMOID);

    /* ################# Update Input Gate ################# */
    if (isCIFGenabled) {
        auto constNode = createConstNode(elementType, f_t->get_shape(), convertToVector(1.f));
        // Couple input with forget gate: 1 - i_f
        i_t = sub(constNode, f_t);

        // TODO: Implement proper scratchBuffer
        // Creating a dummy scratchBuffer with same size as i_t, initialized to 0.0f
        // and then multiplying with i_t so that it gets connected to the graph
        // Then it's concat'ed on axis 1 to make that dim 3x
        scratchBuffer = createConstNode(elementType, i_t->get_shape(), convertToVector(0.f));
        scratchBuffer = std::make_shared<ngraph::opset3::Multiply>(scratchBuffer, i_t);
        std::vector<ngraph::Output<ngraph::Node>> inputs;
        for (int i = 0; i < 3; i++) inputs.push_back(scratchBuffer);
        scratchBuffer = std::make_shared<ngraph::opset3::Concat>(inputs, 1);
    } else {
        if (isLayerNormUsed) {
            i_t_Mean = calculatemean(i_t, reduceAxes, false);
            i_t_Variance = calculateVariance(i_t, i_t_Mean, reduceAxes, elementType);
            i_t = LayerNorm(i_t, i_t_Mean, i_t_Variance, input_layer_norm_weights, input_gate_bias);
        } else {
            // W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}C_{t-1} + b_i
            i_t = add(i_t, input_gate_bias);
        }
        // sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}C_{t-1} + b_i)
        i_t = applyActivation(i_t, ACTIVATION_FUNCTION_SIGMOID);

        // TODO: Implement proper scratchBuffer
        // Creating a dummy scratchBuffer with same size as i_t, initialized to 0.0f
        // and then multiplying with i_t so that it gets connected to the graph
        // Then it's concat'ed on axis 1 to make that dim 4x
        scratchBuffer = createConstNode(elementType, i_t->get_shape(), convertToVector(0.f));
        scratchBuffer = std::make_shared<ngraph::opset3::Multiply>(scratchBuffer, i_t);
        std::vector<ngraph::Output<ngraph::Node>> inputs;
        for (int i = 0; i < 4; i++) inputs.push_back(scratchBuffer);
        scratchBuffer = std::make_shared<ngraph::opset3::Concat>(inputs, 1);
    }

    /* ################# Update Cell Gate ################# */

    if (isLayerNormUsed) {
        c_t_Mean = calculatemean(c_t, reduceAxes, false);
        c_t_Variance = calculateVariance(c_t, c_t_Mean, reduceAxes, elementType);
        c_t = LayerNorm(c_t, c_t_Mean, c_t_Variance, cell_layer_norm_weights, cell_bias);
    } else {
        // W_{xc}x_t+W_{hc}h_{t-1}+b_c
        c_t = add(c_t, cell_bias);
    }
    // g(W_{xc}x_t+W_{hc}h_{t-1}+b_c)
    c_t = applyActivation(c_t, activationFn);

    // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, initial_cell_state), mul(i_t, c_t));
    // clip(ft (.) Ct-1 + it (.) ct, t_{cell})
    C = clip(C, cell_state_clipping);

    /* ################# Update Output Gate ################# */

    // W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t
    o_t = add(o_t, mul(cell2output_weights, C));

    if (isLayerNormUsed) {
        o_t_Mean = calculatemean(o_t, reduceAxes, false);
        o_t_Variance = calculateVariance(o_t, o_t_Mean, reduceAxes, elementType);
        o_t = LayerNorm(o_t, o_t_Mean, o_t_Variance, output_layer_norm_weights, output_gate_bias);
    } else {
        // W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o
        o_t = add(o_t, output_gate_bias);
    }

    // sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o)
    o_t = applyActivation(o_t, ACTIVATION_FUNCTION_SIGMOID);

    std::shared_ptr<ngraph::Node> H;
    if (isProjectionUsed) {
        // o_t odot g(C_t)
        auto dotProd = mul(o_t, applyActivation(C, activationFn));
        // W_{proj}(o_t odot g(C_t))
        auto projWeightsProduct = matMul(projection_weights, dotProd, false, true);
        // W_{proj}(o_t odot g(C_t))+b_{proj}
        auto projBiasAdd = add(transpose(NC_CN, projWeightsProduct), projection_bias);
        // clip(W_{proj}(o_t odot g(C_t))+b_{proj}, t_{proj})
        H = clip(projBiasAdd, proj_clipping);
    } else {
        // o_t odot g(C_t)
        H = mul(o_t, applyActivation(C, activationFn));
    }

    std::vector<std::shared_ptr<ngraph::Node>> LstmOutputs(4, nullptr);
    LstmOutputs[0] = scratchBuffer;
    LstmOutputs[1] = H;
    LstmOutputs[2] = C;
    LstmOutputs[3] = H;

    for (int i = 0; i < 4; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        mNgraphNodes->setOutputAtOperandIndex(outputIndex, LstmOutputs[i]);
        const auto op = sModelInfo->getOperand(outputIndex);
        if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            addResultNode(outputIndex, LstmOutputs[i]);
        }
    }
    return nullptr;
}

std::shared_ptr<ngraph::Node> LSTM::add(const ngraph::Output<ngraph::Node>& lhs,
                                        const ngraph::Output<ngraph::Node>& rhs) {
    return {make_shared<ngraph::opset3::Add>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> LSTM::sub(const ngraph::Output<ngraph::Node>& lhs,
                                        const ngraph::Output<ngraph::Node>& rhs) {
    return {make_shared<ngraph::opset3::Subtract>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
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
            return std::make_shared<ngraph::opset3::Tanh>(arg);
    }
}

std::shared_ptr<ngraph::Node> LSTM::get_num_elements(
    const ngraph::Output<ngraph::Node>& value, const ngraph::Output<ngraph::Node>& reduction_axes) {
    const auto value_shape = std::make_shared<ngraph::opset3::ShapeOf>(value);
    const auto indices = createConstNode(ngraph::element::i32, {}, convertToVector(1));
    const auto dim_values =
        std::make_shared<ngraph::opset3::Gather>(value_shape, indices, reduction_axes);

    return std::make_shared<ngraph::opset3::ReduceProd>(dim_values, reduction_axes);
}

std::shared_ptr<ngraph::Node> LSTM::calculateVariance(
    const ngraph::Output<ngraph::Node>& input, const std::shared_ptr<ngraph::Node>& mean,
    const std::shared_ptr<ngraph::Node>& reduction_axes, const ngraph::element::Type& elementType) {
    // x_i - mean_i
    auto diff = std::make_shared<ngraph::opset3::Subtract>(input, mean);
    // (x_i - mean_i) ** 2
    auto multiply = mul(diff, diff);
    // sum((x_i - mean_i) ** 2)
    auto sum_diff = std::make_shared<ngraph::opset3::ReduceSum>(multiply, reduction_axes, false);
    auto N = get_num_elements(input, reduction_axes);
    N = std::make_shared<ngraph::opset3::Convert>(N, elementType);
    // sum((x_i - mean_i) ** 2) / k
    return std::make_shared<ngraph::opset3::Divide>(sum_diff, N);
}

std::shared_ptr<ngraph::Node> LSTM::calculatemean(
    const ngraph::Output<ngraph::Node>& value, const std::shared_ptr<ngraph::Node>& reduction_axes,
    bool keep_dims) {
    std::shared_ptr<ngraph::Node> elems_number;
    auto value_elem_type = value.get_element_type();
    // sum(x_i[j] for j in range(k))
    auto value_elems_sum =
        std::make_shared<ngraph::opset3::ReduceSum>(value, reduction_axes, keep_dims);
    elems_number = get_num_elements(value, reduction_axes);
    elems_number = std::make_shared<ngraph::opset3::Convert>(elems_number, value_elem_type);
    // sum(x_i[j] for j in range(k)) / k
    return std::make_shared<ngraph::opset3::Divide>(value_elems_sum, elems_number);
}

std::shared_ptr<ngraph::Node> LSTM::LayerNorm(
    const ngraph::Output<ngraph::Node>& input, const std::shared_ptr<ngraph::Node>& mean,
    const std::shared_ptr<ngraph::Node>& variance,
    const std::shared_ptr<ngraph::Node>& normalizedweights,
    const std::shared_ptr<ngraph::Node>& bias) {
    // LayerNormalization
    auto elementeType = ngraph::element::f32;
    auto normalizationConstant = createConstNode(elementeType, {}, convertToVector(1e-8f));

    auto constNode = createConstNode(elementeType, {}, convertToVector(1.0));
    // (x_i - mean_i)
    auto sub_input_mean = sub(input, mean);
    // var_i + epsilon
    auto add_var = add(variance, normalizationConstant);
    // sqrt(var_i + epsilon)
    auto sqrt = std::make_shared<ngraph::opset3::Sqrt>(add_var);
    // 1 / sqrt(var_i + epsilon)
    auto stddev_inv = std::make_shared<ngraph::opset3::Divide>(constNode, sqrt);
    // (x_i - mean_i) * (1 / sqrt(var_i + epsilon))
    auto normalized_input = mul(sub_input_mean, stddev_inv);
    // x_i_normalized * gamma
    auto mul_norm_weights = mul(normalized_input, normalizedweights);
    // x_i_normalized * gamma + beta
    auto output = add(mul_norm_weights, bias);

    return output;
}

bool LSTM::isValidInputTensor(uint32_t inputIndex) {
    const auto& dims = getInputOperandDimensions(inputIndex);
    if (dims.empty()) return false;

    if (dims[0] == 0) return false;

    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
