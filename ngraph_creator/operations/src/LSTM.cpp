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
    inputNode = getInputNode<float>(0);
    const auto& elementType = inputNode->get_element_type();

    // W_{xi}, W_{xf}, W_{xc}, W_{xo}
    if (isCIFGenabled) {
        if (!isCifgDimsEmpty) removeInputNode(1);
    } else {
        input2input_weights = getInputNode<float>(1);
    }
    input2forget_weights = getInputNode<float>(2);
    input2cell_weights = getInputNode<float>(3);
    input2output_weights = getInputNode<float>(4);

    // W_{hi}, W_{hf}, W_{hc}, W_{ho}
    if (isCIFGenabled) {
        if (!isCifgDimsEmpty) removeInputNode(5);
    } else {
        recurrent2input_weights = getInputNode<float>(5);
    }
    recurrent2forget_weights = getInputNode<float>(6);
    recurrent2cell_weights = getInputNode<float>(7);
    recurrent2output_weights = getInputNode<float>(8);

    // W_{ci}, W_{cf}, W_{co}
    if (isPeepholeUsed) {
        if (isCIFGenabled)
            cell2input_weights = createConstNode(elementType, ngraph::Shape{num_units});
        else
            cell2input_weights = getInputNode<float>(9);
        cell2forget_weights = getInputNode<float>(10);
        cell2output_weights = getInputNode<float>(11);
    } else {
        cell2input_weights = createConstNode(elementType, ngraph::Shape{num_units});
        cell2forget_weights = createConstNode(elementType, ngraph::Shape{num_units});
        cell2output_weights = createConstNode(elementType, ngraph::Shape{num_units});
    }

    // b_i, b_f, b_c, b_o
    if (isCIFGenabled) {
        if (!isCifgDimsEmpty) removeInputNode(12);
    } else {
        input_gate_bias = getInputNode<float>(12);
    }
    forget_gate_bias = getInputNode<float>(13);
    cell_bias = getInputNode<float>(14);
    output_gate_bias = getInputNode<float>(15);

    // W_{proj}, b_{proj}
    if (isProjectionUsed) {
        projection_weights = getInputNode<float>(16);
        if (isValidInputTensor(17))
            projection_bias = getInputNode<float>(17);
        else
            projection_bias = createConstNode(elementType, ngraph::Shape{output_size});
    }

    initial_hidden_state = getInputNode<float>(18);  // h_{t-1}
    initial_cell_state = getInputNode<float>(19);    // C_{t-1}

    activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 20);
    cell_state_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 21);

    if (isProjectionUsed)
        proj_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 22);

    std::shared_ptr<ngraph::Node> i_t, f_t, c_t, o_t;
    std::shared_ptr<ngraph::Node> scratchBuffer;

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

    if (isLayerNormUsed) {
        std::shared_ptr<ngraph::Node> input_layer_norm_weights, forget_layer_norm_weights,
            cell_layer_norm_weights, output_layer_norm_weights;

        if (isCIFGenabled)
            input_layer_norm_weights = createConstNode(elementType, ngraph::Shape{num_units});
        else
            input_layer_norm_weights = getInputNode<float>(23);
        forget_layer_norm_weights = getInputNode<float>(24);
        cell_layer_norm_weights = getInputNode<float>(25);
        output_layer_norm_weights = getInputNode<float>(26);

        const auto& elementType = inputNode->get_element_type();
        auto reduceAxes = ngraph::op::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});

        std::shared_ptr<ngraph::Node> i_t_Mean, i_t_Variance, f_t_Mean, f_t_Variance, c_t_Mean,
            c_t_Variance, o_t_Mean, o_t_Variance;

        if (!isCIFGenabled) {
            i_t_Mean = std::make_shared<ngraph::opset3::ReduceMean>(i_t, reduceAxes, false);
            i_t_Variance = calculateVariance(i_t, i_t_Mean, reduceAxes, elementType);
        }
        f_t_Mean = std::make_shared<ngraph::opset3::ReduceMean>(f_t, reduceAxes, false);
        f_t_Variance = calculateVariance(f_t, f_t_Mean, reduceAxes, elementType);

        c_t_Mean = std::make_shared<ngraph::opset3::ReduceMean>(c_t, reduceAxes, false);
        c_t_Variance = calculateVariance(c_t, c_t_Mean, reduceAxes, elementType);

        o_t_Mean = std::make_shared<ngraph::opset3::ReduceMean>(o_t, reduceAxes, false);
        o_t_Variance = calculateVariance(o_t, o_t_Mean, reduceAxes, elementType);

        double normalizationConstant = 1e-8f;
        std::shared_ptr<ngraph::Node> batchNorm_i_t, batchNorm_f_t, batchNorm_c_t, batchNorm_o_t;

        if (!isCIFGenabled)
            batchNorm_i_t = std::make_shared<ngraph::opset3::BatchNormInference>(
                i_t, input_layer_norm_weights, input_gate_bias, i_t_Mean, i_t_Variance,
                normalizationConstant);
        batchNorm_f_t = std::make_shared<ngraph::opset3::BatchNormInference>(
            f_t, forget_layer_norm_weights, forget_gate_bias, f_t_Mean, f_t_Variance,
            normalizationConstant);
        batchNorm_c_t = std::make_shared<ngraph::opset3::BatchNormInference>(
            c_t, cell_layer_norm_weights, cell_bias, c_t_Mean, c_t_Variance, normalizationConstant);
        batchNorm_o_t = std::make_shared<ngraph::opset3::BatchNormInference>(
            o_t, output_layer_norm_weights, output_gate_bias, o_t_Mean, o_t_Variance,
            normalizationConstant);

        if (!isCIFGenabled) i_t = batchNorm_i_t;
        f_t = batchNorm_f_t;
        c_t = batchNorm_c_t;
        o_t = batchNorm_o_t;
    } else {
        // adding bias to gates
        if (!isCIFGenabled) i_t = add(i_t, input_gate_bias);
        f_t = add(f_t, forget_gate_bias);
        c_t = add(c_t, cell_bias);
        o_t = add(o_t, output_gate_bias);
    }

    // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    f_t = applyActivation(clip(f_t, cell_state_clipping), ACTIVATION_FUNCTION_SIGMOID);

    if (isCIFGenabled) {
        // Couple input with forget gate: 1 - i_f
        i_t =
            sub(ngraph::op::Constant::create(f_t->get_element_type(), f_t->get_shape(),
                                             std::vector<float>(shape_size(f_t->get_shape()), 1.f)),
                f_t);

        // TODO: Implement proper scratchBuffer
        // Creating a dummy scratchBuffer with same size as i_t, initialized to 0.0f
        // and then multiplying with i_t so that it gets connected to the graph
        // Then it's concat'ed on axis 1 to make that dim 3x
        scratchBuffer = std::make_shared<ngraph::opset3::Constant>(
            inputNode->get_element_type(), i_t->get_shape(), std::vector<float>{0.f});
        scratchBuffer = std::make_shared<ngraph::opset3::Multiply>(
            scratchBuffer, i_t, ngraph::op::AutoBroadcastType::NUMPY);
        std::vector<ngraph::Output<ngraph::Node>> inputs;
        for (int i = 0; i < 3; i++) inputs.push_back(scratchBuffer);
        scratchBuffer = std::make_shared<ngraph::opset3::Concat>(inputs, 1);
    } else {
        // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        i_t = applyActivation(clip(i_t, cell_state_clipping), ACTIVATION_FUNCTION_SIGMOID);

        // TODO: Implement proper scratchBuffer
        // Creating a dummy scratchBuffer with same size as i_t, initialized to 0.0f
        // and then multiplying with i_t so that it gets connected to the graph
        // Then it's concat'ed on axis 1 to make that dim 4x
        scratchBuffer = std::make_shared<ngraph::opset3::Constant>(
            inputNode->get_element_type(), i_t->get_shape(), std::vector<float>{0.f});
        scratchBuffer = std::make_shared<ngraph::opset3::Multiply>(
            scratchBuffer, i_t, ngraph::op::AutoBroadcastType::NUMPY);
        std::vector<ngraph::Output<ngraph::Node>> inputs;
        for (int i = 0; i < 4; i++) inputs.push_back(scratchBuffer);
        scratchBuffer = std::make_shared<ngraph::opset3::Concat>(inputs, 1);
    }

    // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, initial_cell_state),
                 mul(i_t, applyActivation(clip(c_t, cell_state_clipping), activationFn)));  // C_t

    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = applyActivation(clip(add(o_t, mul(cell2output_weights, C)), cell_state_clipping),
                          ACTIVATION_FUNCTION_SIGMOID);  // o_t

    std::shared_ptr<ngraph::Node> H;
    if (isProjectionUsed) {
        auto projWeightsProduct = matMul(
            projection_weights,
            mul(o_t, applyActivation(clip(C, cell_state_clipping), activationFn)), false, true);
        // clip(W_{proj}(o_t odot g(C_t))+b_{proj}, t_{proj})
        H = clip(add(transpose(NC_CN, projWeightsProduct), projection_bias), proj_clipping);  // h_t
    } else {
        // ot (.) h(Ct)
        H = mul(o_t, applyActivation(clip(C, cell_state_clipping), activationFn));  // h_t
    }

    std::vector<std::shared_ptr<ngraph::Node>> LstmOutputs(4, nullptr);
    LstmOutputs[0] = scratchBuffer;
    LstmOutputs[1] = H;
    LstmOutputs[2] = C;
    LstmOutputs[3] = H;

    // eleminating scratchBuffer, otherwise its crashing loadNetwork()
    for (int i = 0; i < 4; i++) {
        auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, i);
        mNgraphNodes->setOutputAtOperandIndex(outputIndex, LstmOutputs[i]);
        const auto op = sModelInfo->getOperand(outputIndex);
        if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            addResultNode(outputIndex, LstmOutputs[i]);
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
            return std::make_shared<ngraph::opset3::Sigmoid>(arg);
    }
}

std::shared_ptr<ngraph::Node> LSTM::get_num_elements(
    const ngraph::Output<ngraph::Node>& value, const ngraph::Output<ngraph::Node>& reduction_axes) {
    const auto value_shape = std::make_shared<ngraph::opset3::ShapeOf>(value);
    const auto dim_values = std::make_shared<ngraph::opset3::Gather>(
        value_shape, reduction_axes,
        ngraph::opset3::Constant::create(ngraph::element::i64, {}, {0}));

    return std::make_shared<ngraph::opset3::ReduceProd>(
        dim_values, ngraph::opset3::Constant::create(ngraph::element::i64, {}, {0}));
}

std::shared_ptr<ngraph::Node> LSTM::calculateVariance(
    const ngraph::Output<ngraph::Node>& input, const std::shared_ptr<ngraph::Node>& mean,
    const std::shared_ptr<ngraph::Node>& reduction_axes, const ngraph::element::Type& elementType) {
    ngraph::Output<ngraph::Node> diff = std::make_shared<ngraph::opset3::Subtract>(
        input, mean, ngraph::op::AutoBroadcastType::NUMPY);
    diff = std::make_shared<ngraph::opset3::ReduceSum>(
        std::make_shared<ngraph::opset3::Multiply>(diff, diff), reduction_axes, false);
    auto N = get_num_elements(input, reduction_axes);
    N = std::make_shared<ngraph::opset3::Convert>(N, elementType);
    return std::make_shared<ngraph::opset3::Divide>(diff, N, ngraph::op::AutoBroadcastType::NUMPY);
}

std::shared_ptr<ngraph::Node> LSTM::createConstNode(const ngraph::element::Type& elementType,
                                                    const ngraph::Shape& shape) {
    return ngraph::op::Constant::create(elementType, shape, {0});
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
