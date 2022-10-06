#pragma once

#include <Driver.h>
#include <android/log.h>
#include <log/log.h>
#include <NgraphHelper.hpp>
#include <NgraphNodes.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>

#include "ModelManager.h"

#undef LOG_TAG
#define LOG_TAG "OperationBase"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsBase {
protected:
    enum ConversionType {
        NHWC_NCHW,
        NCHW_NHWC,
        IHWO_OIHW,
        OHWI_OIHW,
        NHWC_CWHN,
        CWHN_NHWC,
        NHC_NCH,
        NCH_NHC,
        CNH_NHC,
        NCH_HNC,
        HNC_NCH,
        NHC_CNH,
        BTS_TBS,
        NHCW_NHWC,
        NC_CN
    };
    uint32_t mDefaultOutputIndex;
    uint32_t mDefaultInputIndex = 0;
    int mNnapiOperationIndex;
    std::shared_ptr<ov::Node> transpose(ConversionType type, ov::Output<ov::Node> input);
    virtual std::shared_ptr<ov::Node> createNode() = 0;
    // override createNodeForPlugin in case sPluginType specific implementation is required
    virtual std::shared_ptr<ov::Node> createNodeForPlugin();
    void addResultNode(size_t index, std::shared_ptr<ov::Node> resultNode);

    // helper functions
    bool checkOperandType(uint32_t operandIndex, const int32_t expectedOperandType,
                          const std::string& strLogInfo = "Operand");
    bool checkOutputOperandType(uint32_t index, const int32_t expectedOperandType);
    bool checkInputOperandType(uint32_t index, const int32_t expectedOperandType);
    const vec<uint32_t> getInputOperandDimensions(uint32_t inputIndex);
    bool isValidInputTensor(uint32_t inputIndex);

    std::shared_ptr<ov::Node> getInputNode(uint32_t inputIndex, bool dequantize = true) {
        std::shared_ptr<ov::Node> input;
        auto operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, inputIndex);
        auto operandType = sModelInfo->getOperandType(operandIndex);
        if (sModelInfo->isOperandLifeTimeConst(operandIndex)) {
            auto operandDims = getInputOperandDimensions(inputIndex);
            ov::element::Type elementType;
            switch (operandType) {
                case OperandType::TENSOR_FLOAT32: {
                    elementType = ov::element::f32;
                    auto operandValues = sModelInfo->GetConstVecOperand<float>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                case OperandType::TENSOR_INT32: {
                    elementType = ov::element::i32;
                    auto operandValues = sModelInfo->GetConstVecOperand<int>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                case OperandType::TENSOR_BOOL8: {
                    elementType = ov::element::boolean;
                    auto operandValues = sModelInfo->GetConstVecOperand<uint8_t>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                case OperandType::TENSOR_QUANT8_ASYMM: {
                    elementType = ov::element::u8;
                    auto operandValues = sModelInfo->GetConstVecOperand<uint8_t>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                case OperandType::TENSOR_QUANT8_SYMM:
                case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
                case OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
                    elementType = ov::element::i8;
                    auto operandValues = sModelInfo->GetConstVecOperand<int8_t>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                case OperandType::TENSOR_FLOAT16: {
                    elementType = ov::element::f16;
                    auto operandValues = sModelInfo->GetConstVecOperand<_Float16>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                case OperandType::TENSOR_QUANT16_SYMM: {
                    elementType = ov::element::i16;
                    auto operandValues = sModelInfo->GetConstVecOperand<int16_t>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                case OperandType::TENSOR_QUANT16_ASYMM: {
                    elementType = ov::element::u16;
                    auto operandValues = sModelInfo->GetConstVecOperand<uint16_t>(operandIndex);
                    input = createConstNode(elementType, toNgraphShape(operandDims), operandValues);
                    break;
                }
                default: {
                    ALOGE("Unsupported Tensor type %s inputIndex %d, operandType %d", __func__,
                          inputIndex, operandType);
                    return nullptr;
                }
            }

        } else {
            input = mNgraphNodes->getOperationOutput(operandIndex).get_node_shared_ptr();
        }

        if (dequantize) {
            if (operandType == OperandType::TENSOR_QUANT8_ASYMM ||
                operandType == OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL ||
                operandType == OperandType::TENSOR_QUANT8_ASYMM_SIGNED ||
                operandType == OperandType::TENSOR_QUANT8_SYMM ||
                operandType == OperandType::TENSOR_QUANT16_SYMM ||
                operandType == OperandType::TENSOR_QUANT16_ASYMM) {
                input = DequantizeNode(input, operandIndex, ov::element::f32);
            }
        }

        return input;
    }
    // remove null input node parameter
    void removeInputNode(uint32_t inputIndex) {
        auto operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, inputIndex);
        auto nodeName = mNgraphNodes->getNodeName(operandIndex);
        mNgraphNodes->removeInputParameter(nodeName, operandIndex);
    }

    template <typename T>
    std::shared_ptr<ov::Node> createConstNode(ov::element::Type elementType, ov::Shape shape,
                                              std::vector<T> vals) {
        return std::make_shared<ov::opset8::Constant>(elementType, shape, vals);
    }

    template <typename T>
    std::vector<T> convertToVector(T val) {
        std::vector<T> vec;
        vec.push_back(val);

        return vec;
    }

    std::shared_ptr<ov::Node> addFakeQuantizeNode(std::shared_ptr<ov::Node> inputNode,
                                                  uint32_t index);
    std::shared_ptr<ov::Node> QuantizeNode(std::shared_ptr<ov::Node> input, size_t index,
                                           ov::element::Type quantizeType);
    std::shared_ptr<ov::Node> DequantizeNode(std::shared_ptr<ov::Node> input, uint32_t index,
                                             ov::element::Type dequantizeType);

    const Operand& getInputOperand(uint32_t index) {
        auto inputIdx = sModelInfo->getOperationInput(mNnapiOperationIndex, index);
        return sModelInfo->getOperand(inputIdx);
    }

    const Operand& getOutputOperand(uint32_t index) {
        auto outputIdx = sModelInfo->getOperationOutput(mNnapiOperationIndex, index);
        return sModelInfo->getOperand(outputIdx);
    }

    bool isZeroSizedInput(uint32_t index) {
        auto inputIdx = sModelInfo->getOperationInput(mNnapiOperationIndex, index);
        auto operand = sModelInfo->getOperand(inputIdx);
        auto& dims = operand.dimensions;

        if ((dims.size() > 0) && (dims[0] != 0)) return false;

        return true;
    }

public:
    static std::shared_ptr<NnapiModelInfo> sModelInfo;
    static IntelDeviceType sPluginType;
    std::shared_ptr<NgraphNodes> mNgraphNodes;
    OperationsBase(int operationIndex);
    void setNgraphNodes(std::shared_ptr<NgraphNodes> nodes);
    bool transposed = false;
    virtual bool validate();
    // override validateForPlugin in case sPluginType specific implementation is required
    virtual bool validateForPlugin();
    // override connectOperationToGraph in case Operation has multiple outputs
    virtual void connectOperationToGraph();
    virtual ~OperationsBase() {}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
