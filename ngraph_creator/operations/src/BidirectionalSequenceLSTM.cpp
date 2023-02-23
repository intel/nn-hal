#include <BidirectionalSequenceLSTM.hpp>
#undef LOG_TAG
#define LOG_TAG "BidirectionalSequenceLSTM"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

#define ACTIVATION_FUNCTION_NONE 0
#define ACTIVATION_FUNCTION_RELU 1
#define ACTIVATION_FUNCTION_RELU6 3
#define ACTIVATION_FUNCTION_TANH 4
#define ACTIVATION_FUNCTION_SIGMOID 6

BidirectionalSequenceLSTM::BidirectionalSequenceLSTM(int operationIndex)
    : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool BidirectionalSequenceLSTM::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

void BidirectionalSequenceLSTM::connectOperationToGraph() { createNode(); }

// TODO: Fix VTS issues: in which node value is null/empty, ideally this has to be handled at run
// time, but in nnhal we are generating graph during initialization phase, hence it's not possible
// to do null data check during graph creation. Should explore other alternative possibilities to
// make conditional graph creation at compilation phase.
std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    bool isFwCIFGenabled = false, isFwPeepholeUsed = false, isFwProjectionUsed = false,
         isFwLayerNormUsed = false, isFwCifgDimsEmpty = true;
    bool isBwCIFGenabled = false, isBwPeepholeUsed = false, isBwProjectionUsed = false,
         isBwLayerNormUsed = false, isBwCifgDimsEmpty = true;
    bool hasAuxInputs = false, hasParallelLinking = false;

    // checking if Fw CIFG enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 1) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 5) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 12)) {
        isFwCIFGenabled = true;
    } else {
        if (isValidInputTensor(1) && isValidInputTensor(5) && isValidInputTensor(12))
            isFwCIFGenabled = false;
        else
            isFwCIFGenabled = true;
    }

    // checking if Fw peephole enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 9) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 10) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 11)) {
        isFwPeepholeUsed = false;
    } else {
        if (!isFwCIFGenabled && !isValidInputTensor(9) && isValidInputTensor(10) &&
            isValidInputTensor(11)) {
            isFwCIFGenabled = true;
            isFwCifgDimsEmpty = false;
        }
        if (isFwCIFGenabled) {
            if (isValidInputTensor(10) && isValidInputTensor(11))
                isFwPeepholeUsed = true;
            else
                isFwPeepholeUsed = false;
        } else {
            if (isValidInputTensor(9) && isValidInputTensor(10) && isValidInputTensor(11))
                isFwPeepholeUsed = true;
            else
                isFwPeepholeUsed = false;
        }
    }

    // checking if Fw projection enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 16)) {
        isFwProjectionUsed = false;
    } else {
        if (isValidInputTensor(16))
            isFwProjectionUsed = true;
        else
            isFwProjectionUsed = false;
    }

    // checking if Bw CIFG enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 18) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 22) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 26)) {
        isBwCIFGenabled = true;
    } else {
        if (isValidInputTensor(18) && isValidInputTensor(22) && isValidInputTensor(26))
            isBwCIFGenabled = false;
        else
            isBwCIFGenabled = true;
    }

    // checking if Bw peephole enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 26) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 27) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 28)) {
        isBwPeepholeUsed = false;
    } else {
        if (!isBwCIFGenabled && !isValidInputTensor(26) && isValidInputTensor(27) &&
            isValidInputTensor(28)) {
            isBwCIFGenabled = true;
            isBwCifgDimsEmpty = false;
        }
        if (isBwCIFGenabled) {
            if (isValidInputTensor(27) && isValidInputTensor(28))
                isBwPeepholeUsed = true;
            else
                isBwPeepholeUsed = false;
        } else {
            if (isValidInputTensor(26) && isValidInputTensor(27) && isValidInputTensor(28))
                isBwPeepholeUsed = true;
            else
                isBwPeepholeUsed = false;
        }
    }

    // checking if Bw projection enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 33)) {
        isBwProjectionUsed = false;
    } else {
        if (isValidInputTensor(33))
            isBwProjectionUsed = true;
        else
            isBwProjectionUsed = false;
    }

    // checking if Fw layer normalization enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 53) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 54) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 55) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 56)) {
        isFwLayerNormUsed = false;
    } else {
        if (isFwCIFGenabled) {
            if (isValidInputTensor(54) && isValidInputTensor(55) && isValidInputTensor(56))
                isFwLayerNormUsed = true;
            else
                isFwLayerNormUsed = false;

        } else {
            if (isValidInputTensor(53) && isValidInputTensor(54) && isValidInputTensor(55) &&
                isValidInputTensor(56))
                isFwLayerNormUsed = true;
            else
                isFwLayerNormUsed = false;
        }
    }

    // checking if Bw layer normalization enabled
    if (sModelInfo->isOmittedInput(mNnapiOperationIndex, 57) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 58) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 59) &&
        sModelInfo->isOmittedInput(mNnapiOperationIndex, 60)) {
        isBwLayerNormUsed = false;
    } else {
        if (isBwCIFGenabled) {
            if (isValidInputTensor(58) && isValidInputTensor(59) && isValidInputTensor(60))
                isBwLayerNormUsed = true;
            else
                isBwLayerNormUsed = false;

        } else {
            if (isValidInputTensor(57) && isValidInputTensor(58) && isValidInputTensor(59) &&
                isValidInputTensor(60))
                isBwLayerNormUsed = true;
            else
                isBwLayerNormUsed = false;
        }
    }

    std::shared_ptr<ngraph::Node> inputNode, auxInput;
    std::shared_ptr<ngraph::Node> fw_input2input_weights, fw_input2forget_weights,
        fw_input2cell_weights, fw_input2output_weights, fw_recurrent2input_weights,
        fw_recurrent2forget_weights, fw_recurrent2cell_weights, fw_recurrent2output_weights,
        fw_cell2input_weights, fw_cell2forget_weights, fw_cell2output_weights, fw_input_gate_bias,
        fw_forget_gate_bias, fw_cell_bias, fw_output_gate_bias, fw_projection_weights,
        fw_projection_bias, fw_initial_hidden_state, fw_initial_cell_state;
    std::shared_ptr<ngraph::Node> fwAux_input2input_weights, fwAux_input2forget_weights,
        fwAux_input2cell_weights, fwAux_input2output_weights;
    std::shared_ptr<ngraph::Node> fw_input_layer_norm_weights, fw_forget_layer_norm_weights,
        fw_cell_layer_norm_weights, fw_output_layer_norm_weights;

    std::shared_ptr<ngraph::Node> bw_input2input_weights, bw_input2forget_weights,
        bw_input2cell_weights, bw_input2output_weights, bw_recurrent2input_weights,
        bw_recurrent2forget_weights, bw_recurrent2cell_weights, bw_recurrent2output_weights,
        bw_cell2input_weights, bw_cell2forget_weights, bw_cell2output_weights, bw_input_gate_bias,
        bw_forget_gate_bias, bw_cell_bias, bw_output_gate_bias, bw_projection_weights,
        bw_projection_bias, bw_initial_hidden_state, bw_initial_cell_state;
    std::shared_ptr<ngraph::Node> bwAux_input2input_weights, bwAux_input2forget_weights,
        bwAux_input2cell_weights, bwAux_input2output_weights;
    std::shared_ptr<ngraph::Node> bw_input_layer_norm_weights, bw_forget_layer_norm_weights,
        bw_cell_layer_norm_weights, bw_output_layer_norm_weights;

    const auto& fw_initial_hidden_state_dims = getInputOperandDimensions(35);
    const auto& fw_initial_cell_state_dims = getInputOperandDimensions(36);

    auto fw_num_units = fw_initial_cell_state_dims[1];
    auto fw_output_size = fw_initial_hidden_state_dims[1];

    const auto& bw_initial_hidden_state_dims = getInputOperandDimensions(37);
    const auto& bw_initial_cell_state_dims = getInputOperandDimensions(38);

    auto bw_num_units = bw_initial_cell_state_dims[1];
    auto bw_output_size = bw_initial_hidden_state_dims[1];

    uint32_t activationFn;
    float cell_state_clipping, proj_clipping;

    // Creating input nodes
    inputNode = getInputNode(0);
    const auto& elementType = inputNode->get_element_type();

    /* ########### Forward direction ########### */
    // W_{xi}, W_{xf}, W_{xc}, W_{xo}
    if (isFwCIFGenabled) {
        if (!isFwCifgDimsEmpty) removeInputNode(1);
    } else {
        fw_input2input_weights = getInputNode(1);
    }
    fw_input2forget_weights = getInputNode(2);
    fw_input2cell_weights = getInputNode(3);
    fw_input2output_weights = getInputNode(4);

    // W_{hi}, W_{hf}, W_{hc}, W_{ho}
    if (isFwCIFGenabled) {
        if (!isFwCifgDimsEmpty) removeInputNode(5);
    } else {
        fw_recurrent2input_weights = getInputNode(5);
    }
    fw_recurrent2forget_weights = getInputNode(6);
    fw_recurrent2cell_weights = getInputNode(7);
    fw_recurrent2output_weights = getInputNode(8);

    // W_{ci}, W_{cf}, W_{co}
    if (isFwPeepholeUsed) {
        if (isFwCIFGenabled)
            fw_cell2input_weights =
                createConstNode(elementType, ngraph::Shape{fw_num_units}, convertToVector(0));
        else
            fw_cell2input_weights = getInputNode(9);
        fw_cell2forget_weights = getInputNode(10);
        fw_cell2output_weights = getInputNode(11);
    } else {
        fw_cell2input_weights =
            createConstNode(elementType, ngraph::Shape{fw_num_units}, convertToVector(0));
        fw_cell2forget_weights =
            createConstNode(elementType, ngraph::Shape{fw_num_units}, convertToVector(0));
        fw_cell2output_weights =
            createConstNode(elementType, ngraph::Shape{fw_num_units}, convertToVector(0));
    }

    // b_i, b_f, b_c, b_o
    if (isFwCIFGenabled) {
        if (!isFwCifgDimsEmpty) removeInputNode(12);
    } else {
        fw_input_gate_bias = getInputNode(12);
    }
    fw_forget_gate_bias = getInputNode(13);
    fw_cell_bias = getInputNode(14);
    fw_output_gate_bias = getInputNode(15);

    // W_{proj}, b_{proj}
    if (isFwProjectionUsed) {
        fw_projection_weights = getInputNode(16);
        if (isValidInputTensor(17))
            fw_projection_bias = getInputNode(17);
        else
            fw_projection_bias =
                createConstNode(elementType, ngraph::Shape{fw_output_size}, convertToVector(0));
    }

    fw_initial_hidden_state = getInputNode(35);  // h_{t-1}
    fw_initial_cell_state = getInputNode(36);    // C_{t-1}

    if (isFwLayerNormUsed) {
        if (!isFwCIFGenabled) fw_input_layer_norm_weights = getInputNode(53);
        fw_forget_layer_norm_weights = getInputNode(54);
        fw_cell_layer_norm_weights = getInputNode(55);
        fw_output_layer_norm_weights = getInputNode(56);
    }

    if (!isValidInputTensor(39)) {
        hasAuxInputs = true;
        auxInput = getInputNode(9);
    }

    if (hasAuxInputs) {
        if (!isFwCIFGenabled) fwAux_input2input_weights = getInputNode(40);
        fwAux_input2forget_weights = getInputNode(41);
        fwAux_input2cell_weights = getInputNode(42);
        fwAux_input2output_weights = getInputNode(43);
    }

    /* ########### Backward direction ########### */

    // W_{xi}, W_{xf}, W_{xc}, W_{xo}
    if (isBwCIFGenabled) {
        if (!isBwCifgDimsEmpty) removeInputNode(18);
    } else {
        bw_input2input_weights = getInputNode(18);
    }
    bw_input2forget_weights = getInputNode(19);
    bw_input2cell_weights = getInputNode(20);
    bw_input2output_weights = getInputNode(21);

    // W_{hi}, W_{hf}, W_{hc}, W_{ho}
    if (isBwCIFGenabled) {
        if (!isBwCifgDimsEmpty) removeInputNode(22);
    } else {
        bw_recurrent2input_weights = getInputNode(22);
    }
    bw_recurrent2forget_weights = getInputNode(23);
    bw_recurrent2cell_weights = getInputNode(24);
    bw_recurrent2output_weights = getInputNode(25);

    // W_{ci}, W_{cf}, W_{co}
    if (isBwPeepholeUsed) {
        if (isBwCIFGenabled)
            bw_cell2input_weights =
                createConstNode(elementType, ngraph::Shape{bw_num_units}, convertToVector(0));
        else
            bw_cell2input_weights = getInputNode(26);
        bw_cell2forget_weights = getInputNode(27);
        bw_cell2output_weights = getInputNode(28);
    } else {
        bw_cell2input_weights =
            createConstNode(elementType, ngraph::Shape{bw_num_units}, convertToVector(0));
        bw_cell2forget_weights =
            createConstNode(elementType, ngraph::Shape{bw_num_units}, convertToVector(0));
        bw_cell2output_weights =
            createConstNode(elementType, ngraph::Shape{bw_num_units}, convertToVector(0));
    }

    // b_i, b_f, b_c, b_o
    if (isBwCIFGenabled) {
        if (!isBwCifgDimsEmpty) removeInputNode(29);
    } else {
        bw_input_gate_bias = getInputNode(29);
    }
    bw_forget_gate_bias = getInputNode(30);
    bw_cell_bias = getInputNode(31);
    bw_output_gate_bias = getInputNode(32);

    // W_{proj}, b_{proj}
    if (isBwProjectionUsed) {
        bw_projection_weights = getInputNode(33);
        if (isValidInputTensor(34))
            bw_projection_bias = getInputNode(34);
        else
            bw_projection_bias =
                createConstNode(elementType, ngraph::Shape{bw_output_size}, convertToVector(0));
    }

    bw_initial_hidden_state = getInputNode(37);  // h_{t-1}
    bw_initial_cell_state = getInputNode(38);    // C_{t-1}

    if (isBwLayerNormUsed) {
        if (!isBwCIFGenabled) bw_input_layer_norm_weights = getInputNode(57);
        bw_forget_layer_norm_weights = getInputNode(58);
        bw_cell_layer_norm_weights = getInputNode(59);
        bw_output_layer_norm_weights = getInputNode(60);
    }

    if (hasAuxInputs) {
        if (!isBwCIFGenabled) bwAux_input2input_weights = getInputNode(44);
        bwAux_input2forget_weights = getInputNode(45);
        bwAux_input2cell_weights = getInputNode(46);
        bwAux_input2output_weights = getInputNode(47);
    }

    activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 48);

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT16)) {
        cell_state_clipping = sModelInfo->ParseOperationInput<_Float16>(mNnapiOperationIndex, 49);
        if (isFwProjectionUsed)
            proj_clipping = sModelInfo->ParseOperationInput<_Float16>(mNnapiOperationIndex, 50);
    } else {
        cell_state_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 49);
        if (isFwProjectionUsed)
            proj_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 50);
    }

    auto mergeOutputs = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 51);
    auto isTimeMajor = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 52);

    if (hasAuxInputs) {
        if (!isBwCIFGenabled && !isFwCIFGenabled) {
            if (!isValidInputTensor(40) && !isValidInputTensor(41) && !isValidInputTensor(42) &&
                !isValidInputTensor(43) && !isValidInputTensor(44) && !isValidInputTensor(45) &&
                !isValidInputTensor(46) && !isValidInputTensor(47))
                hasParallelLinking = true;
        } else {
            if (!isValidInputTensor(41) && !isValidInputTensor(42) && !isValidInputTensor(43) &&
                !isValidInputTensor(45) && !isValidInputTensor(46) && !isValidInputTensor(47))
                hasParallelLinking = true;
        }
    }

    const auto& inDims = getInputOperandDimensions(0);
    uint32_t maxTime;

    if (!isTimeMajor) {
        inputNode = transpose(BTS_TBS, inputNode);
        if (hasAuxInputs) {
            auxInput = transpose(BTS_TBS, auxInput);
        }
    }

    if (isTimeMajor) {
        maxTime = inDims[0];
    } else {
        maxTime = inDims[1];
    }

    auto axisNode = createConstNode(ngraph::element::i32, {}, convertToVector(0));
    auto numSplits = maxTime;

    std::vector<ngraph::Output<ngraph::Node>> inputSplit, auxInputSplit;

    inputSplit = std::make_shared<ngraph::opset3::Split>(inputNode, axisNode, numSplits)->outputs();

    if (hasAuxInputs) {
        auxInputSplit =
            std::make_shared<ngraph::opset3::Split>(auxInput, axisNode, numSplits)->outputs();
    }

    std::vector<std::shared_ptr<ngraph::Node>> fw_output_at_each_timestep(maxTime);
    std::vector<std::shared_ptr<ngraph::Node>> bw_output_at_each_timestep(maxTime);
    std::shared_ptr<ngraph::Node> fw_c_lastTimestep, bw_c_lastTimestep;
    std::shared_ptr<ngraph::Node> fw_h_lastTimestep, bw_h_lastTimestep;

    for (uint32_t i = 0; i < maxTime; i++) {
        std::shared_ptr<ngraph::Node> i_t, f_t, c_t, o_t;
        auto dims = createConstNode(ngraph::element::i32, {0}, std::vector<int64_t>{});
        inputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(inputSplit[i], dims);

        // i_t = W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}
        if (!isFwCIFGenabled)
            i_t = add(add(matMul(inputSplit[i], fw_input2input_weights, false, true),
                          matMul(fw_initial_hidden_state, fw_recurrent2input_weights, false, true)),
                      mul(fw_cell2input_weights, fw_initial_cell_state));
        // f_t = W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}
        f_t = add(add(matMul(inputSplit[i], fw_input2forget_weights, false, true),
                      matMul(fw_initial_hidden_state, fw_recurrent2forget_weights, false, true)),
                  mul(fw_cell2forget_weights, fw_initial_cell_state));
        // c_t = W_{xc}x_t+W_{hc}h_{t-1}
        c_t = add(matMul(inputSplit[i], fw_input2cell_weights, false, true),
                  matMul(fw_initial_hidden_state, fw_recurrent2cell_weights, false, true));
        // o_t = W_{xo}x_t+W_{ho}h_{t-1}
        o_t = add(matMul(inputSplit[i], fw_input2output_weights, false, true),
                  matMul(fw_initial_hidden_state, fw_recurrent2output_weights, false, true));

        if (hasAuxInputs) {
            auxInputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(auxInputSplit[i], dims);
            // aux_input * aux_input_weights
            if (!isFwCIFGenabled) {
                auto it_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                    auxInputSplit[i], fwAux_input2input_weights, false, true);
                i_t = add(i_t, it_aux_mul);
            }
            auto ft_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                auxInputSplit[i], fwAux_input2forget_weights, false, true);
            auto ct_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                auxInputSplit[i], fwAux_input2cell_weights, false, true);
            auto ot_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                auxInputSplit[i], fwAux_input2output_weights, false, true);
            f_t = add(f_t, ft_aux_mul);
            c_t = add(c_t, ct_aux_mul);
            o_t = add(o_t, ot_aux_mul);
        }

        /* ################# Update Forget Gate ################# */
        if (isFwLayerNormUsed) {
            f_t = LayerNorm(f_t, fw_forget_layer_norm_weights, fw_forget_gate_bias);
        } else {
            // W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}C_{t-1} + b_f
            f_t = add(f_t, fw_forget_gate_bias);
        }
        // sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}C_{t-1} + b_f)
        f_t = applyActivation(f_t, ACTIVATION_FUNCTION_SIGMOID);

        /* ################# Update Input Gate ################# */
        if (isFwCIFGenabled) {
            auto constNode = createConstNode(elementType, f_t->get_shape(), convertToVector(1.f));
            // Couple input with forget gate: 1 - i_f
            i_t = sub(constNode, f_t);
        } else {
            if (isFwLayerNormUsed) {
                i_t = LayerNorm(i_t, fw_input_layer_norm_weights, fw_input_gate_bias);
            } else {
                // W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}C_{t-1} + b_i
                i_t = add(i_t, fw_input_gate_bias);
            }
            // sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}C_{t-1} + b_i)
            i_t = applyActivation(i_t, ACTIVATION_FUNCTION_SIGMOID);
        }

        /* ################# Update Cell Gate ################# */

        if (isFwLayerNormUsed) {
            c_t = LayerNorm(c_t, fw_cell_layer_norm_weights, fw_cell_bias);
        } else {
            // W_{xc}x_t+W_{hc}h_{t-1}+b_c
            c_t = add(c_t, fw_cell_bias);
        }
        // g(W_{xc}x_t+W_{hc}h_{t-1}+b_c)
        c_t = applyActivation(c_t, activationFn);

        // ft (.) Ct-1 + it (.) ct
        auto C = add(mul(f_t, fw_initial_cell_state), mul(i_t, c_t));
        // clip(ft (.) Ct-1 + it (.) ct, t_{cell})
        C = clip(C, cell_state_clipping);

        /* ################# Update Output Gate ################# */

        // W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t
        o_t = add(o_t, mul(fw_cell2output_weights, C));

        if (isFwLayerNormUsed) {
            o_t = LayerNorm(o_t, fw_output_layer_norm_weights, fw_output_gate_bias);
        } else {
            // W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o
            o_t = add(o_t, fw_output_gate_bias);
        }

        // sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o)
        o_t = applyActivation(o_t, ACTIVATION_FUNCTION_SIGMOID);

        std::shared_ptr<ngraph::Node> H;
        if (isFwProjectionUsed) {
            // o_t odot g(C_t)
            auto dotProd = mul(o_t, applyActivation(C, activationFn));
            // W_{proj}(o_t odot g(C_t))
            auto projWeightsProduct = matMul(fw_projection_weights, dotProd, false, true);
            // W_{proj}(o_t odot g(C_t))+b_{proj}
            auto projBiasAdd = add(transpose(NC_CN, projWeightsProduct), fw_projection_bias);
            // clip(W_{proj}(o_t odot g(C_t))+b_{proj}, t_{proj})
            H = clip(projBiasAdd, proj_clipping);
        } else {
            // o_t odot g(C_t)
            H = mul(o_t, applyActivation(C, activationFn));
        }

        fw_initial_hidden_state = H;
        fw_initial_cell_state = C;
        fw_output_at_each_timestep[i] = H;
        if (i == maxTime - 1) {
            fw_c_lastTimestep = C;
            fw_h_lastTimestep = H;
        }
    }

    for (int i = maxTime - 1; i >= 0; --i) {
        std::shared_ptr<ngraph::Node> i_t, f_t, c_t, o_t;
        auto dims = createConstNode(ngraph::element::i32, {0}, std::vector<int64_t>{});
        std::shared_ptr<ngraph::Node> curStepInput;

        if (hasParallelLinking) {
            curStepInput = std::make_shared<ngraph::opset3::Squeeze>(auxInputSplit[i], dims);
        } else {
            curStepInput = std::make_shared<ngraph::opset3::Squeeze>(inputSplit[i], dims);
        }

        // i_t = W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}
        if (!isBwCIFGenabled)
            i_t = add(add(matMul(curStepInput, bw_input2input_weights, false, true),
                          matMul(bw_initial_hidden_state, bw_recurrent2input_weights, false, true)),
                      mul(bw_cell2input_weights, bw_initial_cell_state));
        // f_t = W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}
        f_t = add(add(matMul(curStepInput, bw_input2forget_weights, false, true),
                      matMul(bw_initial_hidden_state, bw_recurrent2forget_weights, false, true)),
                  mul(bw_cell2forget_weights, bw_initial_cell_state));
        // c_t = W_{xc}x_t+W_{hc}h_{t-1}
        c_t = add(matMul(curStepInput, bw_input2cell_weights, false, true),
                  matMul(bw_initial_hidden_state, bw_recurrent2cell_weights, false, true));
        // o_t = W_{xo}x_t+W_{ho}h_{t-1}
        o_t = add(matMul(curStepInput, bw_input2output_weights, false, true),
                  matMul(bw_initial_hidden_state, bw_recurrent2output_weights, false, true));

        if (hasAuxInputs && !hasParallelLinking) {
            auxInputSplit[i] = std::make_shared<ngraph::opset3::Squeeze>(auxInputSplit[i], dims);
            // aux_input * aux_input_weights
            if (!isBwCIFGenabled) {
                auto it_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                    auxInputSplit[i], bwAux_input2input_weights, false, true);
                i_t = add(i_t, it_aux_mul);
            }
            auto ft_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                auxInputSplit[i], bwAux_input2forget_weights, false, true);
            auto ct_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                auxInputSplit[i], bwAux_input2cell_weights, false, true);
            auto ot_aux_mul = std::make_shared<ngraph::opset3::MatMul>(
                auxInputSplit[i], bwAux_input2output_weights, false, true);
            f_t = add(f_t, ft_aux_mul);
            c_t = add(c_t, ct_aux_mul);
            o_t = add(o_t, ot_aux_mul);
        }

        /* ################# Update Forget Gate ################# */
        if (isBwLayerNormUsed) {
            f_t = LayerNorm(f_t, bw_forget_layer_norm_weights, bw_forget_gate_bias);
        } else {
            // W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}C_{t-1} + b_f
            f_t = add(f_t, bw_forget_gate_bias);
        }
        // sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}C_{t-1} + b_f)
        f_t = applyActivation(f_t, ACTIVATION_FUNCTION_SIGMOID);

        /* ################# Update Input Gate ################# */
        if (isBwCIFGenabled) {
            auto constNode = createConstNode(elementType, f_t->get_shape(), convertToVector(1.f));
            // Couple input with forget gate: 1 - i_f
            i_t = sub(constNode, f_t);
        } else {
            if (isBwLayerNormUsed) {
                i_t = LayerNorm(i_t, bw_input_layer_norm_weights, bw_input_gate_bias);
            } else {
                // W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}C_{t-1} + b_i
                i_t = add(i_t, bw_input_gate_bias);
            }
            // sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}C_{t-1} + b_i)
            i_t = applyActivation(i_t, ACTIVATION_FUNCTION_SIGMOID);
        }

        /* ################# Update Cell Gate ################# */

        if (isBwLayerNormUsed) {
            c_t = LayerNorm(c_t, bw_cell_layer_norm_weights, bw_cell_bias);
        } else {
            // W_{xc}x_t+W_{hc}h_{t-1}+b_c
            c_t = add(c_t, bw_cell_bias);
        }
        // g(W_{xc}x_t+W_{hc}h_{t-1}+b_c)
        c_t = applyActivation(c_t, activationFn);

        // ft (.) Ct-1 + it (.) ct
        auto C = add(mul(f_t, bw_initial_cell_state), mul(i_t, c_t));
        // clip(ft (.) Ct-1 + it (.) ct, t_{cell})
        C = clip(C, cell_state_clipping);

        /* ################# Update Output Gate ################# */

        // W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t
        o_t = add(o_t, mul(bw_cell2output_weights, C));

        if (isBwLayerNormUsed) {
            o_t = LayerNorm(o_t, bw_output_layer_norm_weights, bw_output_gate_bias);
        } else {
            // W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o
            o_t = add(o_t, bw_output_gate_bias);
        }

        // sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o)
        o_t = applyActivation(o_t, ACTIVATION_FUNCTION_SIGMOID);

        std::shared_ptr<ngraph::Node> H;
        if (isBwProjectionUsed) {
            // o_t odot g(C_t)
            auto dotProd = mul(o_t, applyActivation(C, activationFn));
            // W_{proj}(o_t odot g(C_t))
            auto projWeightsProduct = matMul(bw_projection_weights, dotProd, false, true);
            // W_{proj}(o_t odot g(C_t))+b_{proj}
            auto projBiasAdd = add(transpose(NC_CN, projWeightsProduct), bw_projection_bias);
            // clip(W_{proj}(o_t odot g(C_t))+b_{proj}, t_{proj})
            H = clip(projBiasAdd, proj_clipping);
        } else {
            // o_t odot g(C_t)
            H = mul(o_t, applyActivation(C, activationFn));
        }

        bw_initial_hidden_state = H;
        bw_initial_cell_state = C;
        bw_output_at_each_timestep[i] = H;
        if (i == 0) {
            bw_c_lastTimestep = C;
            bw_h_lastTimestep = H;
        }
    }

    std::shared_ptr<ngraph::Node> fwOutputNode, bwOutputNode;
    std::vector<uint32_t> fwShape, bwShape;

    /* ########### Forward direction ########### */
    fwOutputNode = std::make_shared<ngraph::opset3::Concat>(fw_output_at_each_timestep, 0);

    auto fwOutput_batch = fwOutputNode->get_shape()[0] / maxTime;
    fwShape.push_back(maxTime);
    fwShape.push_back(fwOutput_batch);
    fwShape.push_back(fwOutputNode->get_shape()[1]);

    auto fwShapeNode = createConstNode(ngraph::element::i32,
                                       ngraph::Shape{inputNode->get_shape().size()}, fwShape);

    fwOutputNode = std::make_shared<ngraph::opset3::Reshape>(fwOutputNode, fwShapeNode, false);

    /* ########### Backward direction ########### */
    bwOutputNode = std::make_shared<ngraph::opset3::Concat>(bw_output_at_each_timestep, 0);

    auto bwOutput_batch = bwOutputNode->get_shape()[0] / maxTime;
    bwShape.push_back(maxTime);
    bwShape.push_back(bwOutput_batch);
    bwShape.push_back(bwOutputNode->get_shape()[1]);

    auto bwShapeNode = createConstNode(ngraph::element::i32,
                                       ngraph::Shape{inputNode->get_shape().size()}, bwShape);

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
    if (fwOp.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
        addResultNode(fwOutputIndex, fwOutputNode);
        ALOGD("%s Add result %d", __func__, fwOutputIndex);
    }

    if (!mergeOutputs) {
        auto bwOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 1);
        mNgraphNodes->setOutputAtOperandIndex(bwOutputIndex, bwOutputNode);
        ALOGD("%s Set Output index %d", __func__, bwOutputIndex);
        const auto bwOp = sModelInfo->getOperand(bwOutputIndex);
        if (bwOp.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(bwOutputIndex, bwOutputNode);
            ALOGD("%s Add result %d", __func__, bwOutputIndex);
        }
    }

    const auto& outputsSize = sModelInfo->getOperationOutputsSize(mNnapiOperationIndex);

    if (outputsSize > 2) {
        int fw_activation_state_op_index, fw_cell_state_op_index;
        int bw_activation_state_op_index, bw_cell_state_op_index;

        fw_activation_state_op_index = sModelInfo->getOperationOutput(mNnapiOperationIndex, 2);
        mNgraphNodes->setOutputAtOperandIndex(fw_activation_state_op_index, fw_c_lastTimestep);
        ALOGD("%s Set Output index %d", __func__, fw_activation_state_op_index);
        const auto fw_activation_state_op = sModelInfo->getOperand(fw_activation_state_op_index);
        if (fw_activation_state_op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(fw_activation_state_op_index, fw_c_lastTimestep);
            ALOGD("%s Add result %d", __func__, fw_activation_state_op_index);
        }

        fw_cell_state_op_index = sModelInfo->getOperationOutput(mNnapiOperationIndex, 3);
        mNgraphNodes->setOutputAtOperandIndex(fw_cell_state_op_index, fw_h_lastTimestep);
        ALOGD("%s Set Output index %d", __func__, fw_cell_state_op_index);
        const auto fw_cell_state_op = sModelInfo->getOperand(fw_cell_state_op_index);
        if (fw_cell_state_op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(fw_cell_state_op_index, fw_h_lastTimestep);
            ALOGD("%s Add result %d", __func__, fw_cell_state_op_index);
        }

        bw_activation_state_op_index = sModelInfo->getOperationOutput(mNnapiOperationIndex, 4);
        mNgraphNodes->setOutputAtOperandIndex(bw_activation_state_op_index, bw_c_lastTimestep);
        ALOGD("%s Set Output index %d", __func__, bw_activation_state_op_index);
        const auto bw_activation_state_op = sModelInfo->getOperand(bw_activation_state_op_index);
        if (bw_activation_state_op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(bw_activation_state_op_index, bw_c_lastTimestep);
            ALOGD("%s Add result %d", __func__, bw_activation_state_op_index);
        }

        bw_cell_state_op_index = sModelInfo->getOperationOutput(mNnapiOperationIndex, 5);
        mNgraphNodes->setOutputAtOperandIndex(bw_cell_state_op_index, bw_h_lastTimestep);
        ALOGD("%s Set Output index %d", __func__, bw_cell_state_op_index);
        const auto bw_cell_state_op = sModelInfo->getOperand(bw_cell_state_op_index);
        if (bw_cell_state_op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
            addResultNode(bw_cell_state_op_index, bw_h_lastTimestep);
            ALOGD("%s Add result %d", __func__, bw_cell_state_op_index);
        }
    }

    return nullptr;
}

std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::add(
    const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs) {
    return {std::make_shared<ngraph::opset3::Add>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::sub(
    const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs) {
    return {
        std::make_shared<ngraph::opset3::Subtract>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::mul(
    const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs) {
    return {
        std::make_shared<ngraph::opset3::Multiply>(lhs, rhs, ngraph::op::AutoBroadcastType::NUMPY)};
}

std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::matMul(
    const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs,
    bool transpose_lhs, bool transpose_rhs) {
    return {std::make_shared<ngraph::opset3::MatMul>(lhs, rhs, transpose_lhs, transpose_rhs)};
}

std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::clip(
    const ngraph::Output<ngraph::Node>& data, float m_clip) const {
    if (m_clip == 0.f) {
        return data.get_node_shared_ptr();
    }
    return std::make_shared<ngraph::opset3::Clamp>(data, -m_clip, m_clip);
}
std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::applyActivation(
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

std::shared_ptr<ngraph::Node> BidirectionalSequenceLSTM::LayerNorm(
    const ngraph::Output<ngraph::Node>& input,
    const std::shared_ptr<ngraph::Node>& normalizedweights,
    const std::shared_ptr<ngraph::Node>& bias) {
    // LayerNormalization
    auto normalizationConstant =
        createConstNode(input.get_element_type(), {}, convertToVector(1e-8f));
    auto axis = ngraph::op::Constant::create(ngraph::element::i32, {}, {-1});
    auto mean = std::make_shared<ngraph::opset3::ReduceMean>(input, axis, true);
    // x_i - mean_i
    auto diff = sub(input, mean);
    // (x_i - mean_i) ** 2
    auto multiply = mul(diff, diff);
    // mean((x_i - mean_i) ** 2)
    auto var = std::make_shared<ngraph::opset3::ReduceMean>(multiply, axis, true);
    // var_i + epsilon
    auto add_var = add(var, normalizationConstant);
    // sqrt(var_i + epsilon)
    auto sqrt = std::make_shared<ngraph::opset3::Sqrt>(add_var);
    // (x_i - mean_i) / sqrt(var_i + epsilon)
    auto stddev_inv = std::make_shared<ngraph::opset3::Divide>(diff, sqrt);
    // x_i_normalized * gamma
    auto mul_norm_weights = mul(stddev_inv, normalizedweights);
    // x_i_normalized * gamma + beta
    auto output = add(mul_norm_weights, bias);

    return output;
}

bool BidirectionalSequenceLSTM::isValidInputTensor(uint32_t inputIndex) {
    const auto& dims = getInputOperandDimensions(inputIndex);
    if (dims.empty()) return false;

    if (dims[0] == 0) return false;

    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
