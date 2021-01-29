#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#define NNLOG1

#include "IRLayers.h"
#ifdef NNLOG1
#define LOG_TAG "CreateNgraph"
#include <android/log.h>
#include <log/log.h>
#define LOGDIMS(d, header)                                                           \
    do {                                                                             \
        auto size = (d).size();                                                      \
        ALOGD("%s: vectors {%d, %d, %d, %d}", header, (d)[0], size > 1 ? (d)[1] : 0, \
              size > 2 ? (d)[2] : 0, size > 3 ? (d)[3] : 0);                         \
    } while (0)
#endif

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
class CreateNgraph {
private:
    std::map<std::string, std::shared_ptr<ngraph::Node>> mNodes;
    ngraph::ParameterVector mInputParams;
    std::shared_ptr<ngraph::Node> mLastNode;
    std::vector<std::shared_ptr<ngraph::Node>> mResultNodes;

public:
    InferenceEngine::CNNNetwork generate(std::string xmlPath, std::string binPath) {
#ifdef NNLOG1
        ALOGD("%s : Called with xmlPath %s", __func__, xmlPath.c_str());
#endif
        auto ngraph_function = std::make_shared<ngraph::Function>(mResultNodes, mInputParams);
        InferenceEngine::CNNNetwork cnn = InferenceEngine::CNNNetwork(ngraph_function);
        try {
            cnn.serialize(xmlPath, binPath);
        } catch (const std::exception& ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
        }
        return cnn;
    }
    void addNode(std::string nodeName, std::shared_ptr<ngraph::Node> node) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s adding node : %s", __func__, nodeName.c_str(),
              node->get_name().c_str());
#endif
        mLastNode = node;
        mNodes[nodeName] = node;
    }
    std::string getNodeName(std::string nodeName) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s ", __func__, nodeName.c_str());
#endif
        return mNodes[nodeName]->get_name();
    }
    void setResultNode(std::string nodeName) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s ", __func__, nodeName.c_str());
#endif
        // Transpose to NHWC for the Result node
        ngraph::AxisVector order{0, 2, 3, 1};
        const auto order_node = ngraph::opset3::Constant::create(
            ngraph::element::i64, ngraph::Shape{order.size()}, order);
        std::string tempNodeName = nodeName + "_preNode";
        mNodes[tempNodeName] = mNodes[nodeName];
        auto transpose =
            std::make_shared<ngraph::opset3::Transpose>(mNodes[tempNodeName], order_node);
        addNode(nodeName, transpose);
        mResultNodes.push_back(mNodes[nodeName]);
    }
    void addInputParameter(std::string nodeName, std::vector<size_t> shape) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s ", __func__, nodeName.c_str());
        LOGDIMS(shape, __func__);
#endif
        std::shared_ptr<ngraph::opset3::Parameter> input =
            std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape(shape));
        mInputParams.push_back(input);
        addNode(nodeName, input);
    }
    void addClamp(std::string nodeName, std::string inputName, const double min, const double max) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s inputName=%s ", __func__, nodeName.c_str(), inputName.c_str());
#endif
        addNode(nodeName, std::make_shared<ngraph::opset3::Clamp>(mNodes[inputName], min, max));
    }
    void addRelu(std::string nodeName, std::string inputName) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s inputName=%s ", __func__, nodeName.c_str(), inputName.c_str());
#endif
        addNode(nodeName, std::make_shared<ngraph::opset3::Relu>(mNodes[inputName]));
    }
    void addReshape(std::string nodeName, std::string inputName, std::vector<size_t> shape) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s inputName=%s ", __func__, nodeName.c_str(), inputName.c_str());
#endif
        try {
            // Pre Transpose [[ NCHW > NHWC
            ngraph::AxisVector pre_order{0, 2, 3, 1};
            const auto pre_order_node = ngraph::opset3::Constant::create(
                ngraph::element::i64, ngraph::Shape{pre_order.size()}, pre_order);
            std::string preNodeName = nodeName + "_priorTranspose";
            auto pre_transpose =
                std::make_shared<ngraph::opset3::Transpose>(mNodes[inputName], pre_order_node);
            addNode(preNodeName, pre_transpose);
            // Pre Transpose ]]
            auto shapeNode = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{shape.size()}, shape.data());
            std::string currNodeName = nodeName + "_actualReshape";
            addNode(currNodeName, std::make_shared<ngraph::opset3::Reshape>(mNodes[preNodeName],
                                                                            shapeNode, true));

            // Post Transpose [[ NHWC > NCHW
            ngraph::AxisVector post_order{0, 3, 1, 2};
            const auto post_order_node = ngraph::opset3::Constant::create(
                ngraph::element::i64, ngraph::Shape{post_order.size()}, post_order);
            auto post_transpose =
                std::make_shared<ngraph::opset3::Transpose>(mNodes[currNodeName], post_order_node);
            addNode(nodeName,
                    post_transpose);  // while adding a new node connected to this node, inputName
                                      // passed will be the current nodeName. Hence, the Post
                                      // Transpose node should be mapped to it.
            // Post Transpose ]]
        } catch (const std::exception& ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
        }
    }
    void addConcat(std::string nodeName, std::vector<std::string> inputNames, int axis) {
        std::string str;
        for (const auto& name : inputNames) str = str + name + ";";
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s inputNames=%s ", __func__, nodeName.c_str(), str.c_str());
#endif
        std::vector<std::shared_ptr<ngraph::Node>> inputs;
        for (int i = 0; i < inputNames.size(); ++i) {
            inputs.push_back(mNodes[inputNames[i]]);
        }
        try {
            addNode(nodeName, std::make_shared<ngraph::opset3::Concat>(inputs, axis));
        } catch (const std::exception& ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
        }
    }
    void addConvolution(std::string nodeName, std::string inputName, GenConvParams& gPrms) {
#ifdef NNLOG1
        ALOGD("%s : nodeName=%s inputName=%s ", __func__, nodeName.c_str(), inputName.c_str());
        LOGDIMS(gPrms.weightsDims, __func__);
#endif
        std::shared_ptr<ngraph::Node> input;
        try {
            ngraph::Shape constShape = ngraph::Shape(
                &gPrms.weightsDims[0], &gPrms.weightsDims[0] + gPrms.weightsDims.size());
            std::shared_ptr<ngraph::Node> ieWeights = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::f32, constShape, gPrms.weightsBuf);
            if (gPrms.groups != 1) {  // GroupConvolution
                std::vector<size_t> shape(&gPrms.weightsDims[0], &gPrms.weightsDims[0] + 4);
                shape[0] /= gPrms.groups;
                shape.insert(shape.begin(), gPrms.groups);

                auto shapeNode = std::make_shared<ngraph::opset3::Constant>(
                    ngraph::element::i64, ngraph::Shape{shape.size()}, shape.data());
                ieWeights = std::make_shared<ngraph::opset3::Reshape>(ieWeights, shapeNode, true);
            }
            ngraph::op::PadType auto_pad = ngraph::op::PadType::EXPLICIT;
            if (!std::strcmp(gPrms.pad_type, "explicit"))
                auto_pad = ngraph::op::PadType::EXPLICIT;
            else if (!std::strcmp(gPrms.pad_type, "same_upper"))
                auto_pad = ngraph::op::PadType::SAME_UPPER;
            else if (!std::strcmp(gPrms.pad_type, "valid"))
                auto_pad = ngraph::op::PadType::VALID;
            if (gPrms.groups == 1) {
                input = std::make_shared<ngraph::opset3::Convolution>(
                    mNodes[inputName], ieWeights, ngraph::Strides(gPrms.strides),
                    ngraph::CoordinateDiff(gPrms.pads_begin),
                    ngraph::CoordinateDiff(gPrms.pads_end), ngraph::Strides(gPrms.dilations),
                    auto_pad);
            } else {
                input = std::make_shared<ngraph::opset3::GroupConvolution>(
                    mNodes[inputName], ieWeights, ngraph::Strides(gPrms.strides),
                    ngraph::CoordinateDiff(gPrms.pads_begin),
                    ngraph::CoordinateDiff(gPrms.pads_end), ngraph::Strides(gPrms.dilations),
                    auto_pad);
            }
        } catch (const std::exception& ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
        }
        if (gPrms.biasesBuf.size() > 0) {
#ifdef NNLOG1
            ALOGD("%s : nodeName=%s has biases size %d", __func__, nodeName.c_str(),
                  gPrms.biasesBuf.size());
            LOGDIMS(gPrms.biasesDims, __func__);
#endif
            std::vector<size_t> shape(input->get_shape().size(), 1);
            shape[1] = gPrms.biasesDims[0];
            ngraph::Shape constShape = ngraph::Shape(shape);
            std::shared_ptr<ngraph::Node> ieBiases = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::f32, constShape, gPrms.biasesBuf);
            addNode(nodeName + "_hasbiases", input);
            addNode(nodeName, std::make_shared<ngraph::opset3::Add>(
                                  input, ieBiases, ngraph::op::AutoBroadcastType::NUMPY));
        }  // while adding a new node connected to this node, inputName passed will be the current
           // nodeName. Hence, the Add node should be mapped to it.
        else {
            addNode(nodeName, input);
        }
    }
};
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android