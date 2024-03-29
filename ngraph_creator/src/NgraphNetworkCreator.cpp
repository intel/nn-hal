#include <NgraphNetworkCreator.hpp>
#undef LOG_TAG
#define LOG_TAG "NgraphNetworkCreator"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

NgraphNetworkCreator::NgraphNetworkCreator(std::shared_ptr<NnapiModelInfo> modelInfo,
                                           IntelDeviceType deviceType)
    : mModelInfo(modelInfo),
      mNgraphNodes(std::make_shared<NgraphNodes>(mModelInfo->getOperandsSize(),
                                                 mModelInfo->getModelOutputsSize())),
      mOpFactoryInstance(deviceType, mModelInfo, mNgraphNodes) {
    auto nnapiOperationsSize = mModelInfo->getOperationsSize();
    mOperationNodes.resize(nnapiOperationsSize);
    for (size_t index = 0; index < nnapiOperationsSize; index++) {
        const auto& nnapiOperationType = mModelInfo->getOperationType(index);
        auto operationNode = mOpFactoryInstance.getOperation(index, nnapiOperationType);
        if (operationNode == nullptr) {
            ALOGV("%s Unsupported Operation type %d", __func__, nnapiOperationType);
        } else
            operationNode->mNgraphNodes = mNgraphNodes;
        mOperationNodes[index] = operationNode;
    }
    ALOGV("%s Constructed", __func__);
}

NgraphNetworkCreator::~NgraphNetworkCreator() { ALOGV("%s Destructed", __func__); }

bool NgraphNetworkCreator::createInputParams() {
    for (auto i : mModelInfo->getModelInputIndexes()) {
        std::shared_ptr<ngraph::opset3::Parameter> inputParam;
        auto& nnapiOperand = mModelInfo->getOperand(i);
        auto& dims = nnapiOperand.dimensions;
        ALOGV("createInputParams operand %d dims.size(%zu)", i, dims.size());
        // keeping this condition to make VTS pass. Operation's optional input lifetime is supposed
        // to be "NO_VALUE"
        // TODO: Remove these checks to support zero_sized input tensors
        if (dims.size() > 0) {
            if (dims[0] != 0) {
                switch (nnapiOperand.type) {
                    case OperandType::FLOAT32:
                    case OperandType::TENSOR_FLOAT32:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::f32, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    case OperandType::INT32:
                    case OperandType::TENSOR_INT32:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::i32, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    case OperandType::BOOL:
                    case OperandType::TENSOR_BOOL8:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::boolean, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    case OperandType::TENSOR_QUANT8_ASYMM:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::u8, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    case OperandType::TENSOR_QUANT8_SYMM:
                    case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
                    case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::i8, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    case OperandType::TENSOR_FLOAT16:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::f16, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    case OperandType::TENSOR_QUANT16_SYMM:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::i16, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    case OperandType::TENSOR_QUANT16_ASYMM:
                        inputParam = std::make_shared<ngraph::opset3::Parameter>(
                            ngraph::element::u16, ngraph::Shape(dims.begin(), dims.end()));
                        ALOGV("createInputParams created inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        break;
                    default:
                        ALOGE("createInputParams Failure at inputIndex %d, type %d", i,
                              nnapiOperand.type);
                        inputParam = nullptr;
                        return false;
                }

                mNgraphNodes->addInputParam(inputParam);
                mNgraphNodes->setOutputAtOperandIndex(i, inputParam);
            } else {
                mNgraphNodes->setInvalidNode(i);
            }
        } else {
            mNgraphNodes->setInvalidNode(i);
        }
    }
    return true;
}

void NgraphNetworkCreator::getSupportedOperations(std::vector<bool>& supportedOperations) {
    for (size_t i = 0; i < mModelInfo->getOperationsSize(); i++) {
        if (!mOperationNodes[i] || !mOperationNodes[i]->validateForPlugin())
            supportedOperations[i] = false;
        else
            supportedOperations[i] = true;
        ALOGD("%s index %zu type %d, supported : %d", __func__, i, mModelInfo->getOperationType(i),
              static_cast<int>(supportedOperations[i]));
    }
}

bool NgraphNetworkCreator::validateOperations() {
    for (size_t i = 0; i < mModelInfo->getOperationsSize(); i++) {
        if (!mOperationNodes[i] || !mOperationNodes[i]->validateForPlugin()) {
            ALOGE("%s index %zu type %d not supported", __func__, i,
                  mModelInfo->getOperationType(i));
            return false;
        }
    }
    return true;
}

bool NgraphNetworkCreator::initializeModel() {
    ALOGV("%s Called", __func__);
    if (!createInputParams()) return false;
    for (size_t i = 0; i < mModelInfo->getOperationsSize(); i++) {
        if (mOperationNodes[i] == nullptr) {
            ALOGE("initializeModel Failure at type %d", mModelInfo->getOperationType(i));
            return false;
        }
        try {
            mOperationNodes[i]->connectOperationToGraph();
        } catch (const std::exception& ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
            return false;
        }
    }
    ALOGD("initializeModel Success");
    return true;
}

const std::string& NgraphNetworkCreator::getNodeName(uint32_t index) {
    ALOGV("getNodeName %d", index);
    return mNgraphNodes->getNodeName(index);
}

std::shared_ptr<ngraph::Function> NgraphNetworkCreator::generateGraph() {
    ALOGV("%s Called", __func__);
    std::shared_ptr<ngraph::Function> ret;
    try {
        if (initializeModel()) ret = mNgraphNodes->generateGraph();
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
    }
    return ret;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
