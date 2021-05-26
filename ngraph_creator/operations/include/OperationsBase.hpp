#pragma once

#include <Driver.h>
#include <android/log.h>
#include <log/log.h>
#include <NgraphHelper.hpp>
#include <NgraphNodes.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>

#include "ModelManager.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsBase {
protected:
    enum ConversionType { NHWC_NCHW, NCHW_NHWC, IHWO_OIHW, OHWI_OIHW, NHC_NCH, NCH_NHC, NC_CN };
    uint32_t mDefaultOutputIndex;
    int mNnapiOperationIndex;
    std::shared_ptr<ngraph::Node> transpose(ConversionType type,
                                            ngraph::Output<ngraph::Node> input);
    virtual std::shared_ptr<ngraph::Node> createNode() = 0;
    // override createNodeForPlugin in case sPluginType specific implementation is required
    virtual std::shared_ptr<ngraph::Node> createNodeForPlugin();
    std::shared_ptr<ngraph::Node> toNCHW(size_t inputIndex, size_t outputIndex);
    void addResultNode(size_t index, std::shared_ptr<ngraph::Node> resultNode);

    // helper functions
    bool checkOperandType(uint32_t operandIndex, const int32_t expectedOperandType,
                          const std::string& strLogInfo = "Operand");
    bool checkOutputOperandType(uint32_t index, const int32_t expectedOperandType);
    bool checkInputOperandType(uint32_t index, const int32_t expectedOperandType);
    const vec<uint32_t> getInputOperandDimensions(uint32_t inputIndex);
    template <typename T>
    std::shared_ptr<ngraph::Node> getInputNode(uint32_t inputIndex) {
        auto operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, inputIndex);
        if (sModelInfo->isOperandLifeTimeConst(operandIndex)) {
            ngraph::element::Type elementType;
            auto operandType = sModelInfo->getOperandType(operandIndex);
            if (operandType == OperandType::TENSOR_FLOAT32) elementType = ngraph::element::f32;
            if (operandType == OperandType::TENSOR_INT32) elementType = ngraph::element::i32;
            if (operandType == OperandType::TENSOR_BOOL8) elementType = ngraph::element::boolean;
            auto operandValues = sModelInfo->GetConstVecOperand<T>(operandIndex);
            auto operandDims = getInputOperandDimensions(inputIndex);
            return std::make_shared<ngraph::opset3::Constant>(
                elementType, toNgraphShape(operandDims), operandValues);
        } else
            return mNgraphNodes->getOperationOutput(operandIndex).get_node_shared_ptr();
    }
    // remove null input node parameter
    void removeInputNode(uint32_t inputIndex) {
        auto operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, inputIndex);
        auto nodeName = mNgraphNodes->getNodeName(operandIndex);
        mNgraphNodes->removeInputParameter(nodeName, operandIndex);
    }

public:
    static std::shared_ptr<NnapiModelInfo> sModelInfo;
    static std::string sPluginType;
    std::shared_ptr<NgraphNodes> mNgraphNodes;
    OperationsBase(int operationIndex);
    void setNgraphNodes(std::shared_ptr<NgraphNodes> nodes);
    virtual bool validate();
    // override connectOperationToGraph in case Operation has multiple outputs
    virtual void connectOperationToGraph();
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android