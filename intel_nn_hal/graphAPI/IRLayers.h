/*
 * INTEL CONFIDENTIAL
 * Copyright 2017 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once
#include "IRLayer.h"
#include "IRDocument.h"
#include "file_utils.h"
#include "ie_common.h"
#include <cassert>
#include "ie_layers_property.hpp"

//#define LOG_TAG "graphAPI"

#ifdef NNLOG
#include <android/log.h>
#include <log/log.h>
#endif

namespace IRBuilder
{

extern int layer_name_count;
extern InferenceEngine::Precision g_layer_precision;

inline OutputPort addOutput(const IRLayer &layer, const InferenceEngine::SizeVector &dims)
{
    std::string d_name = layer->name;
    if(!layer->outData.empty())
    {
        std::stringstream oss;
        oss << d_name << ":" << layer->outData.size();
        d_name = oss.str();
    }
    OutputPort data;
    if(dims.size() == 2)
    {
        std::cout << "addOutput dims size 2"<< std::endl;
        InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::NC);
        data = std::make_shared<InferenceEngine::Data>(d_name, td);

    }
    else if(dims.size() == 4)
    {
        std::cout << "addOutput dims size "<< dims.size()<<std::endl;
        //InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::ANY);
        InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::NCHW);
        //InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::NHWC);
        data = std::make_shared<InferenceEngine::Data>(d_name, td);

    }
    else {
        std::cout << "addOutput dims size "<< dims.size()<<std::endl;
        //InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::ANY);
        InferenceEngine::TensorDesc td(g_layer_precision, dims, InferenceEngine::Layout::C);
        data = std::make_shared<InferenceEngine::Data>(d_name, td);
    }

    layer->outData.push_back(data);
    data->creatorLayer = layer;

    #ifdef NNLOG
    std::vector<size_t> outdims = data->getTensorDesc().getDims();
    for (int i=0; i< outdims.size(); i++) {
      ALOGI("addOutput data dims[%d] = %lu ", i, outdims[i]);
    }
    #endif

    return data;
}


template<typename T>
void addAttr(IRLayer layer, const std::string &a_name, T val)
{
    std::stringstream oss;
    oss << val;
    layer->params[a_name] = oss.str();
};
template<typename T, typename S>
std::shared_ptr<T> As(const std::shared_ptr<S> &src)
{ return /*std::dynamic_pointer_cast<T>(src)*/std::static_pointer_cast<T>(src); }  //aks


/*
* @brief Creates a generic layer with one input and one output
*/

inline IRLayer Generic(const std::string &type) {
    std::string name = type + "-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    prms.type = type;
    return std::make_shared<InferenceEngine::CNNLayer>(prms);
}

inline IRLayer Generic(const std::string &type, const OutputPort &src)
{
    std::string name = type + "-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    auto layer = std::make_shared<InferenceEngine::CNNLayer>(prms);
    layer->type = type;
    src >> layer;
    addOutput(layer, src->getTensorDesc().getDims());
    return layer;
}


inline OutputPort output(const IRLayer &src, int index = 0)
{
    return src->outData[index];
}

inline IRLayer LayerOf(const OutputPort &src)
{
  return src->creatorLayer.lock();
}

inline IRLayer Generic(const std::string &type, const IRLayer &src)
{
    return Generic(type, output(src));
}

template<typename T, typename A>
std::string dumpVec(std::vector<T, A> const &vec)
{
    if(vec.empty()) return "[]";
    std::stringstream oss;
    oss << "[" << vec[0];
    for(size_t i = 1; i < vec.size(); i++) oss << "," << vec[i];
    oss << "]";
    return oss.str();
}

namespace FCLayer
{
static IRLayer create(const IRBlob::Ptr &weights, const OutputPort &src)
{
    #ifdef NNLOG
    ALOGI("Create FC layer");
    #endif
    std::string name = "FC-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;

    //auto inDims = src->getDims(); // (batch, IFM)
    auto inDims = src->getTensorDesc().getDims(); // (batch, IFM)
    //std::cout << "inDims size "<<inDims.size()<< "inDims[0] "<<inDims[0]<< "inDims[1] "<<inDims[1]<< std::endl;


    auto wDim = weights->getTensorDesc().getDims();
    //std::cout << "wDim size "<<wDim.size()<<"wDim[0] "<<wDim[0]<< "wDim[1] "<<wDim[1]<< std::endl;

    IR_ASSERT(inDims.size() == 2);

    unsigned int ofm = 0;
    if (wDim.size() == 2)
    {
        //std::cout << "inDims[1]"<<inDims[1]<< "wDim[1]" <<wDim[1]<< std::endl;

        #ifdef NNLOG
        ALOGI("inDims[0] = %d inDims[1] = %d", inDims[0], inDims[1]);
        ALOGI("wDim[0] = %d wDim[1] = %d", wDim[0], wDim[1]);
        #endif

        IR_ASSERT(inDims[1] == wDim[1]); // Weights: (Out,In)
        ofm = static_cast<unsigned int>(wDim[0]); // Out
    } else if (wDim.size()==1) // linear, just a blob, line in IR
    {
        ofm = static_cast<unsigned int>(weights->size() / inDims[1]);
        IR_ASSERT(inDims[1]*ofm == weights->size()); // should be divided properly
    } else
        THROW_IE_EXCEPTION << "expecting weights for FC only as 1 dim (blob) or 2 dim (Matrix)";


    auto fc = std::make_shared<InferenceEngine::FullyConnectedLayer>(prm);
    fc->type = "FullyConnected";

    fc->_out_num = ofm;
	  addAttr(fc, "out-size ", ofm);  //aks added
    // todo: assert that input should be cols
    addOutput(fc, {inDims[0], static_cast<uint32_t>(fc->_out_num)});
    src >> fc;
    fc->_weights = weights;
    fc->blobs["weights"] = weights; // todo: have setter for those layers...
    return fc;
}
};


inline InferenceEngine::CNNLayer::Ptr operator*(const IRBlob::Ptr &weights, const IRLayer &b)
{
    std::cout << "FCLayer::create operator*(const IRBlob::Ptr &weights, const IRLayer &b)"<< std::endl;
    return FCLayer::create(weights, output(b));
}

inline OutputPort operator*(const IRBlob::Ptr &weights, const OutputPort &op)
{
    std::cout << "FCLayer::create operator*(const IRBlob::Ptr &weights, const OutputPort &op)"<< std::endl;
    return output(FCLayer::create(weights, op));
}

static OutputPort ScaleShiftNode(const OutputPort &src, const IRBlob::Ptr &scale, const IRBlob::Ptr &bias) {
    std::cout << "ScaleShiftNode"<< std::endl;
    std::string name = "ConstMul-";  // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    prm.type = "ScaleShift";
    auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);

    src >> l;
    l->_weights = scale;
    l->_broadcast = false;
    l->_biases = bias;
    l->blobs["biases"] = bias;
    return addOutput(l, src->getTensorDesc().getDims());
}


/*
inline IRLayer AddConst(const IRLayer &lhs, const IRBlob::Ptr &biases)
{
    auto fc = As<InferenceEngine::WeightableLayer>(lhs);
    if (fc) {
      // todo: check if biases was not already being set
      fc->_biases = biases;
      fc->blobs["biases"] = biases;
      return lhs; // it was fused with prev layer
    } else {
	// need to create an add with Const here using ScaleShift with no weights...
	    THROW_IE_EXCEPTION << "not implemented yet" ;
    }

}
*/

inline OutputPort AddTryConst(const OutputPort &src, const IRBlob::Ptr &biases) {
    auto fc = As<InferenceEngine::WeightableLayer>(LayerOf(src));
    if (fc) {
        // todo: check if biases was not lready being set
        std::cout << "AddTryConst"<< std::endl;
        #ifdef NNLOG
        ALOGI("AddTryConst for biases");
        #endif

        fc->_biases = biases;
        fc->blobs["biases"] = biases;
        return src;  // it was fused with prev layer
    } else {
        // need to create an add with Const here using ScaleShift with no weights...
        // there are two options, scale shift with no weights, or cosnt with an Add
        return ScaleShiftNode(src, nullptr, biases);
    }
}


inline OutputPort operator+(const OutputPort &src, const IRBlob::Ptr &biases) {
    return AddTryConst(src, biases);
}
/*
inline OutputPort operator+(const OutputPort &src, const IRBlob::Ptr &biases)
{
    auto l = LayerOf(src);
    return output(AddConst(l, biases));
}
*/

namespace ConvLayer
{
static IRLayer create(const OutputPort &src)
{
    std::string name = "Conv-"; // todo: make it unique
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    name = name << layer_name_count++;
    prm.name = name;
    auto conv_layer = std::make_shared<InferenceEngine::ConvolutionLayer>(prm);
    conv_layer->type = "Convolution";
    src >> conv_layer;
    return conv_layer;
}
};

struct Point2D
{
    int x, y;

    inline int size() const
    { return x * y; }
};
inline Point2D operator+(const Point2D &a, const Point2D &b)
{
    return {a.x + b.x, a.y + b.y};
}
inline Point2D operator-(const Point2D &a, const Point2D &b)
{
    return {a.x - b.x, a.y - b.y};
}
inline Point2D operator*(const Point2D &a, const Point2D &b)
{
    return {a.x * b.x, a.y * b.y};
}
inline Point2D operator/(const Point2D &a, const Point2D &b)
{
    return {a.x / b.x, a.y / b.y};
}
inline Point2D operator+(const Point2D &a, const int &rhs)
{
    return {a.x + rhs, a.y + rhs};
}

struct ConvolutionParams
{
    int groups=1;
    Point2D kernel, stride = {1}, pad_start = {0}, pad_end = {0};
    int num_output_planes;
    IRBlob::Ptr weights;
    IRBlob::Ptr biases;
    std::string padType;
};

inline size_t in_ch(const OutputPort &src)
{
    auto dims = src->getTensorDesc().getDims();
    return dims.size() == 4 ? dims[1] : dims[2];
}
inline OutputPort Convolution(const OutputPort &src, const ConvolutionParams &prms)
{
    auto ret = As<InferenceEngine::ConvolutionLayer>(ConvLayer::create(src));
    auto inDims = src->getTensorDesc().getDims();
    IR_ASSERT(inDims.size() == 4);
    //IR_ASSERT(prms.kernel.size() * n(src) * prms.num_output_planes == prms.weights->size());
	  IR_ASSERT((prms.kernel.size() * in_ch(src) * prms.num_output_planes)/prms.groups == prms.weights->size());

    ret->_weights = prms.weights;
    ret->blobs["weights"] = prms.weights;

    ret->_biases = prms.biases;
    ret->blobs["biases"] = prms.biases;

    ret->_kernel.clear();
    ret->_kernel.insert(InferenceEngine::X_AXIS, prms.kernel.x);
    ret->_kernel.insert(InferenceEngine::Y_AXIS, prms.kernel.y);
    ret->_stride.clear();
    ret->_stride.insert(InferenceEngine::X_AXIS, prms.stride.x);
    ret->_stride.insert(InferenceEngine::Y_AXIS, prms.stride.y);
    ret->_padding.clear();
    ret->_padding.insert(InferenceEngine::X_AXIS, prms.pad_start.x);
    ret->_padding.insert(InferenceEngine::Y_AXIS, prms.pad_start.y);
    ret->_pads_end.clear();
    ret->_pads_end.insert(InferenceEngine::X_AXIS, prms.pad_end.x);
    ret->_pads_end.insert(InferenceEngine::Y_AXIS, prms.pad_end.y);

    ret->_dilation.clear();
    ret->_dilation.insert(InferenceEngine::X_AXIS, 1);
    ret->_dilation.insert(InferenceEngine::Y_AXIS, 1);

    ret->_group = prms.groups;
    ret->_out_depth = prms.num_output_planes;

    //<data dilation-x="1" dilation-y="1" group="1" kernel-x="3" kernel-y="3" output="8" pad-x="0" pad-y="0" stride="1,1,2,2" stride-x="2" stride-y="2"/>

    ret->params["auto_pad"] = prms.padType;
    ret->params["dilation-x"] = std::to_string(ret->_dilation.at(InferenceEngine::X_AXIS));
    ret->params["dilation-y"] = std::to_string(ret->_dilation.at(InferenceEngine::Y_AXIS));
    ret->params["group"] = std::to_string(ret->_group);

    ret->params["kernel-x"] = std::to_string(ret->_kernel.at(InferenceEngine::X_AXIS));
    ret->params["kernel-y"] = std::to_string(ret->_kernel.at(InferenceEngine::Y_AXIS));
    ret->params["output"] = std::to_string(ret->_out_depth);
    ret->params["pad-begin-x"] = std::to_string(ret->_padding.at(InferenceEngine::X_AXIS));
    ret->params["pad-begin-y"] = std::to_string(ret->_padding.at(InferenceEngine::Y_AXIS));
    ret->params["pad-end-x"] = std::to_string(ret->_pads_end.at(InferenceEngine::X_AXIS));
    ret->params["pad-end-y"] = std::to_string(ret->_pads_end.at(InferenceEngine::Y_AXIS));
    ret->params["stride-x"] = std::to_string(ret->_stride.at(InferenceEngine::X_AXIS));
    ret->params["stride-y"] = std::to_string(ret->_stride.at(InferenceEngine::Y_AXIS));


    #ifdef NNLOG
        ALOGI("Convolution  prms.groups = %d kernel.x= %d kernel.y= %d stride.x= %d stride.y= %d pad_start.x= %d pad_start.y= %d \
        pad_end.x= %d pad_end.y= %d ", prms.groups, \
        ret->_kernel.at(InferenceEngine::X_AXIS), ret->_kernel.at(InferenceEngine::Y_AXIS), \
        ret->_stride.at(InferenceEngine::X_AXIS), ret->_stride.at(InferenceEngine::Y_AXIS), \
        ret->_padding.at(InferenceEngine::X_AXIS), ret->_padding.at(InferenceEngine::Y_AXIS), \
        ret->_pads_end.at(InferenceEngine::X_AXIS), ret->_pads_end.at(InferenceEngine::Y_AXIS));
    #endif

    if (prms.padType == "explicit") {
          Point2D in_size = {static_cast<int>(inDims[3]), static_cast<int>(inDims[2])};
          //Point2D out_size = (in_size + prms.pad_start + prms.pad_end - prms.kernel + prms.stride) / prms.stride + 1;
          Point2D out_size = (in_size - prms.kernel + prms.stride + prms.pad_start + prms.pad_end ) / prms.stride;

          addOutput(ret, {inDims[0], (size_t) prms.num_output_planes, (size_t) out_size.y, (size_t) out_size.x}); //nchw
          //addOutput(ret, {inDims[0], (size_t) out_size.y, (size_t) out_size.x, (size_t) prms.num_output_planes}); //nhwc
    }
    else {

          //Calculate output height and width for uneven padding
          size_t inputN = inDims[0];
          size_t IH = inDims[2];
          size_t IW = inDims[3];
          size_t KH = 0, KW = 0;
          float OH_temp, OW_temp;

          if (ret->_dilation[InferenceEngine::Y_AXIS])
              KH = (ret->_kernel[InferenceEngine::Y_AXIS] - 1) * ret->_dilation[InferenceEngine::Y_AXIS] + 1;
          else
              KH = ret->_kernel[InferenceEngine::Y_AXIS];
          if (ret->_dilation[InferenceEngine::X_AXIS])
              KW = (ret->_kernel[InferenceEngine::X_AXIS] - 1) * ret->_dilation[InferenceEngine::X_AXIS] + 1;
          else
              KW = ret->_kernel[InferenceEngine::X_AXIS];

          size_t SH = ret->_stride[InferenceEngine::Y_AXIS];
          size_t SW = ret->_stride[InferenceEngine::X_AXIS];
          size_t PH = ret->_padding[InferenceEngine::Y_AXIS];
          size_t PW = ret->_padding[InferenceEngine::X_AXIS];
          size_t OC = ret->_out_depth;

          if (prms.padType == "valid") {
              OH_temp = std::ceil((IH - KH + 1.f) / SH);
              OW_temp = std::ceil((IW - KW + 1.f) / SW);
          } else if (prms.padType == "same_upper") {
              OH_temp = std::ceil(1.f * IH / SH);
              OW_temp = std::ceil(1.f * IW / SW);
          } else if (prms.padType == "same_lower") {
              OH_temp = std::floor(1.f * IH / SH);
              OW_temp = std::floor(1.f * IW / SW);
          }

          size_t OH = static_cast<size_t>(OH_temp);
          size_t OW = static_cast<size_t>(OW_temp);
          addOutput(ret, {inputN, OC, OH, OW});
    }

    return output(ret);
}
struct BatchNormParams
{
    float epsilon;
    IRBlob::Ptr weights;
    IRBlob::Ptr bias;
};

inline IRLayer BatchNormalization(const OutputPort &src, BatchNormParams &prms)
{
    auto inp = src;
    std::string name = "BatchNormalization-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::BatchNormalizationLayer>(prm);
    l->type = "BatchNormalization";
    src >> l;
    l->epsilon = prms.epsilon;
    l->_weights = prms.weights;
    l->_biases = prms.bias;
    addOutput(l, inp->getTensorDesc().getDims());
    return l;
}

inline OutputPort LRN(const OutputPort &src, float alpha, float beta, int local_size, bool isAcross=true, float k=1)
{
    auto inp = src;
    std::string name = "Norm-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::NormLayer>(prm);
    l->type = "Norm";

    src >> l;
    l->_alpha = alpha;
    l->_beta = beta;
    l->_isAcrossMaps = isAcross;
    l->_size = local_size;
    l->_k = (unsigned int)k;
    return addOutput(l, inp->getTensorDesc().getDims());
}

inline OutputPort Crop(const OutputPort &src,
                         const std::vector<int> &axis,
                         const std::vector<int> &dim,
                         const std::vector<int> &offset)
{
    auto inp = src;
    std::string name = "Crop-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::CropLayer>(prm);
    l->type = "Crop";
    src >> l;
    l->axis = axis;
    l->dim = dim;
    l->offset = offset;
    InferenceEngine::SizeVector sv(dim.begin(), dim.end());
    return addOutput(l, sv);
}

inline OutputPort Pooling(const OutputPort &inp,
                            const Point2D &kernel,
                            const Point2D &stride,
                            const Point2D &pad,
                            InferenceEngine::PoolingLayer::PoolType type)
{
    auto src = inp;
    std::string name = "Pooling-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto ret = std::make_shared<InferenceEngine::PoolingLayer>(prm);
    ret->type = "Pooling";

/*
    ret->_kernel_x = kernel.x;
    ret->_kernel_y = kernel.y;
    ret->_stride_x = stride.x;
    ret->_stride_y = stride.y;
    ret->_padding_x = pad.x;
    ret->_padding_y = pad.y;
*/
    ret->_kernel.clear();
    ret->_kernel.insert(InferenceEngine::X_AXIS, kernel.x);
    ret->_kernel.insert(InferenceEngine::Y_AXIS, kernel.y);
    ret->_stride.clear();
    ret->_stride.insert(InferenceEngine::X_AXIS, stride.x);
    ret->_stride.insert(InferenceEngine::Y_AXIS, stride.y);
    ret->_padding.clear();
    ret->_padding.insert(InferenceEngine::X_AXIS, pad.x);
    ret->_padding.insert(InferenceEngine::Y_AXIS, pad.y);

    ret->_type = type;
    ret->_exclude_pad = true;

    auto inDims = src->getTensorDesc().getDims();

    Point2D in_size = {static_cast<int>(inDims[3]), static_cast<int>(inDims[2])};
    // todo: handle uneven padding
    Point2D out_size = (in_size + pad + pad - kernel + stride) / stride;
    src >> ret;
    addOutput(ret, {inDims[0], inDims[1], (size_t) out_size.y, (size_t) out_size.x});
    return output(ret);
}

inline OutputPort Pooling(const OutputPort &inp,
                            const Point2D &kernel,
                            const Point2D &stride,
                            const Point2D &pad_start,
                            const Point2D &pad_end,
                            std::string padType,
                            InferenceEngine::PoolingLayer::PoolType type)
{
      auto src = inp;
      std::string name = "Pooling-"; // todo: make it unique
      name = name << layer_name_count++;
      InferenceEngine::LayerParams prm;
      prm.precision = g_layer_precision;
      prm.name = name;
      auto ret = std::make_shared<InferenceEngine::PoolingLayer>(prm);
      ret->type = "Pooling";

      ret->_kernel.clear();
      ret->_kernel.insert(InferenceEngine::X_AXIS, kernel.x);
      ret->_kernel.insert(InferenceEngine::Y_AXIS, kernel.y);
      ret->_stride.clear();
      ret->_stride.insert(InferenceEngine::X_AXIS, stride.x);
      ret->_stride.insert(InferenceEngine::Y_AXIS, stride.y);
      ret->_padding.clear();
      ret->_padding.insert(InferenceEngine::X_AXIS, pad_start.x);
      ret->_padding.insert(InferenceEngine::Y_AXIS,  pad_start.y);
      ret->_pads_end.clear();
      ret->_pads_end.insert(InferenceEngine::X_AXIS, pad_end.x);
      ret->_pads_end.insert(InferenceEngine::Y_AXIS, pad_end.y);
      ret->_type = type;
      ret->_exclude_pad = true;

      #ifdef NNLOG
//        ALOGI("Pooling  kernel.x= %d kernel.y= %d stride.x= %d stride.y= %d pad_start.x= %d pad_start.y= %d \
//        pad_end.x= %d pad_end.y= %d ", kernel.x, kernel.y, stride.x, stride.y, pad_start.x, pad_start.y, pad_end.x, pad_end.y);
      #endif
      //<data exclude-pad="true" kernel-x="4" kernel-y="4" pad-x="0" pad-y="0" pool-method="avg" stride="1,1,2,2" stride-x="2" stride-y="2"/>
      ret->params["auto_pad"] = padType;
      ret->params["_exclude_pad"] = std::to_string(ret->_exclude_pad);
      ret->params["kernel-x"] = std::to_string(ret->_kernel.at(InferenceEngine::X_AXIS));
      ret->params["kernel-y"] = std::to_string(ret->_kernel.at(InferenceEngine::Y_AXIS));
      ret->params["pad-begin-x"] = std::to_string(ret->_padding.at(InferenceEngine::X_AXIS));
      ret->params["pad-begin-y"] = std::to_string(ret->_padding.at(InferenceEngine::Y_AXIS));
      ret->params["pad-end-x"] = std::to_string(ret->_pads_end.at(InferenceEngine::X_AXIS));
      ret->params["pad-end-y"] = std::to_string(ret->_pads_end.at(InferenceEngine::Y_AXIS));
      std::string poolingType = ret->_type == InferenceEngine::PoolingLayer::PoolType::AVG ? "avg" : "max";
      ret->params["pool-method"] = poolingType; //std::to_string(poolingType);
      ret->params["stride-x"] = std::to_string(ret->_stride.at(InferenceEngine::X_AXIS));
      ret->params["stride-y"] = std::to_string(ret->_stride.at(InferenceEngine::Y_AXIS));

      src >> ret;

      auto inDims = src->getTensorDesc().getDims();

      if (padType == "explicit") {
          Point2D in_size = {static_cast<int>(inDims[3]), static_cast<int>(inDims[2])};
          // todo: handle uneven padding
          Point2D out_size = (in_size - kernel + pad_start + pad_end + stride) / stride; // add stride-1 to round ceiling

          addOutput(ret, {inDims[0], inDims[1], (size_t) out_size.y, (size_t) out_size.x});
      } else {
      //Calculate output height and width for uneven padding
          float OHTemp = 1.f, OWTemp = 1.f;
          size_t inputN = inDims[0];
          size_t IC = inDims[1];
          size_t IH = inDims[2];
          size_t IW = inDims[3];
          size_t KH = ret->_kernel[InferenceEngine::Y_AXIS];
          size_t KW = ret->_kernel[InferenceEngine::X_AXIS];
          size_t SH = ret->_stride[InferenceEngine::Y_AXIS];
          size_t SW = ret->_stride[InferenceEngine::X_AXIS];
          size_t PH = ret->_padding[InferenceEngine::Y_AXIS];
          size_t PW = ret->_padding[InferenceEngine::X_AXIS];

          if (padType == "valid") {
              OHTemp = std::ceil((IH - KH + 1.f) / SH);
              OWTemp = std::ceil((IW - KW + 1.f) / SW);
          } else if (padType == "same_upper") {
              OHTemp = std::ceil(1.f * IH / SH);
              OWTemp = std::ceil(1.f * IW / SW);
          } else if (padType == "same_lower") {
              OHTemp = std::floor(1.f * IH / SH);
              OWTemp = std::floor(1.f * IW / SW);
          }

          size_t OH = static_cast<size_t>(OHTemp);
          size_t OW = static_cast<size_t>(OWTemp);
          addOutput(ret, {inputN, IC, OH, OW});
      }

      return output(ret);
}


namespace SumLayer
{
   static IRLayer create(const OutputPort &src1, const OutputPort &src2)
   {
       std::string name = "Sum-"; // todo: make it unique
       name = name << layer_name_count++;
       InferenceEngine::LayerParams prm;
       prm.precision = g_layer_precision;
       prm.name = name;
       auto sum = std::make_shared<InferenceEngine::EltwiseLayer>(prm);
       sum->type = "Eltwise";
       src1 >> sum;
       src2 >> sum;
       if(src1->getTensorDesc().getDims() != src2->getTensorDesc().getDims()) THROW_IE_EXCEPTION << "input sizes for Element wise Sum do not match";
       addOutput(sum, src1->getTensorDesc().getDims());
       return sum;
   }
};

namespace MulLayer
{
static IRLayer create(const OutputPort &src1, const OutputPort &src2)
{
    std::string name = "Mul-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto mul = std::make_shared<InferenceEngine::EltwiseLayer>(prm);
    mul->type = "Mul";
    mul->_operation = InferenceEngine::EltwiseLayer::Prod;
    src1 >> mul;
    src2 >> mul;
    if(src1->getTensorDesc().getDims() != src2->getTensorDesc().getDims()) THROW_IE_EXCEPTION << "input sizes for Element wise Mul do not match";
    addOutput(mul, src1->getTensorDesc().getDims());
    return mul;
}
};

inline OutputPort operator*(const OutputPort &a, const OutputPort &b)
{
    return output(MulLayer::create(a, b));
}

namespace ScaleShift
{

static OutputPort Diagnoal(const Vector &weights, const OutputPort &src)
{
    std::string name = "ConstMul-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);
    l->type = "ConstMul";
    src >> l;
    addOutput(l, src->getTensorDesc().getDims());
    l->_weights = weights.data;
    if(weights.length == 1) l->_broadcast = 0;
    else if(weights.length == src->getTensorDesc().getDims()[1])
    { l->_broadcast = 1; }

    return output(l);
}
static InferenceEngine::CNNLayer::Ptr create(OutputPort src,
                                             IRBlob::Ptr scale,
                                             IRBlob::Ptr bias)
{
    std::string name = "ConstMul-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::ScaleShiftLayer>(prm);
    l->type = "ScaleShift";
    src >> l;
    l->_weights = scale;
    l->_broadcast = false;
    addOutput(l, src->getTensorDesc().getDims());
    //AddConst(l, bias);
    return l;
}
};

inline OutputPort operator*(const Vector &weights, const IRLayer &b)
{
    return (ScaleShift::Diagnoal(weights, output(b)));
}

inline OutputPort operator*(const Vector &weights, const OutputPort &op)
{
    return (ScaleShift::Diagnoal(weights, op));
}

namespace ActivationLayer
{
extern const std::string Sigmoid;

extern const std::string Tanh;

extern const std::string ReLU;

static IRLayer create(const OutputPort &src, const std::string &type)
{
    std::string name = type + "-"; // todo: make it unique
    name = name << layer_name_count++;
    IRLayer layer;
    if((strncasecmp(type.c_str(), "relu", type.size()) == 0))
    {
        InferenceEngine::LayerParams prm;
        prm.precision = g_layer_precision;
        prm.name = name;
        layer = std::make_shared<InferenceEngine::ReLULayer>(prm);
        layer->type = "ReLU";
    }
    else if((strncasecmp(type.c_str(), "tanh", type.size()) == 0))
    {
        InferenceEngine::LayerParams prm;
        prm.precision = g_layer_precision;
        prm.name = name;
        layer = std::make_shared<InferenceEngine::TanHLayer>(prm);
        layer->type = "TanH";
    }
    else if((strncasecmp(type.c_str(), "sigmoid", type.size()) == 0))
    {
        InferenceEngine::LayerParams prm;
        prm.precision = g_layer_precision;
        prm.name = name;
        layer = std::make_shared<InferenceEngine::SigmoidLayer>(prm);
        layer->type = "Sigmoid";
    }
    else
    {
        InferenceEngine::LayerParams prm;
        prm.precision = g_layer_precision;
        prm.name = name;
        layer = std::make_shared<InferenceEngine::CNNLayer>(prm);
        layer->type = "Activation";
        addAttr(layer, "type", type);
    }

    src >> layer;

    std::vector<size_t> dims = src->getTensorDesc().getDims();
    #ifdef NNLOG
    for (int i=0; i< dims.size(); i++) {
      ALOGI("Activation function output dims[%d] = %lu ", i, dims[i]);
    }
    #endif

    addOutput(layer, src->getTensorDesc().getDims());
    return layer;
}

static IRLayer create(const IRLayer &src, const std::string &type)
{
    return create(output(src), type);
}

};

template<typename T>
OutputPort ReLU(const T &src)
{
    return output(ActivationLayer::create(src, ActivationLayer::ReLU));
}

template<typename T>
OutputPort Sigmoid(const T &src)
{
    return output(ActivationLayer::create(src, ActivationLayer::Sigmoid));
}

template<typename T>
OutputPort Tanh(const T &src)
{
    return output(ActivationLayer::create(src, ActivationLayer::Tanh));
}

namespace SplitUtil
{

static IRLayer create(int size, const OutputPort &src, int axis = 1)
{
    std::string name = "Split-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto me = std::make_shared<InferenceEngine::SplitLayer>(prm);
    me->type = "Split";
    addAttr(me, "axis", axis);
    src >> me;
    auto out_dim = src->getTensorDesc().getDims();
    // axis = static_cast<int>(out_dim.size()) - axis - 1; // todo: we are all in reverse here :-(
    out_dim[axis] = out_dim[axis] / size;
    IR_ASSERT(out_dim[axis]*size == src->getTensorDesc().getDims()[axis]);

    for(int i = 0; i < size; i++)
    {
        addOutput(me, out_dim);
    }
    return me;
}
};

inline std::vector<OutputPort> Split(const OutputPort &src, int splitElements, int axis = 1)
{
    return SplitUtil::create(splitElements, src, axis)->outData;
}

inline std::vector<OutputPort> Split(const IRLayer &src, int splitElements, int axis = 1)
{
    return Split(output(src), splitElements, axis);
}

inline OutputPort Concat(const std::vector<OutputPort> inputs, int axis = 1)
{
    std::string name = "Concat-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto ret = std::make_shared<InferenceEngine::ConcatLayer>(prm);
    ret->type = "Concat";
    addAttr(ret, "axis", axis);
    inputs[0] >> ret;
    auto outDim = inputs[0]->getTensorDesc().getDims();
    // it was fixed, should be backward compatiobale though...
    // axis = static_cast<int>(outDim.size()) - axis - 1; // todo: we are all in reverse here :-(
    auto axisSize = outDim[axis];
    for(int i = 1; i < inputs.size(); ++i)
    {
        inputs[i] >> ret;
        axisSize += inputs[i]->getTensorDesc().getDims()[axis];
    }
    outDim[axis] = axisSize;
    return addOutput(ret, outDim);
}

//template<typename T>
inline OutputPort Clamp(const OutputPort &src, float min, float max)
{
    std::string name = "Clamp-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    auto layer = std::make_shared<InferenceEngine::ClampLayer>(prms);
    layer->type = "Clamp";
    layer->min_value = min;
    layer->max_value = max;
    layer->params["max"] = std::to_string(layer->max_value);
    layer->params["min"] = std::to_string(layer->min_value);
    src >> layer;
    addOutput(layer, src->getTensorDesc().getDims());
    return output(layer);
}

inline OutputPort L2Normalization(const OutputPort &src, bool isAcross, bool isShareChannel)
{
    auto layer = Generic("Normalize", src);
    addAttr(layer, "across_spatial", isAcross ? 1 : 0);
    addAttr(layer, "channel_shared", isShareChannel ? 1 : 0);
    return output(layer);
}

inline OutputPort Reshape(const TensorDims &newDims, const OutputPort &src)
{
    if(sizeOf(src->getTensorDesc().getDims()) != sizeOf(newDims)) THROW("Cannot reorder different volumes");

/*//first implementation
    if(src->creatorLayer.lock()->type == "Reshape") // fuse reshapes
    {
        src->setDims(newDims);
        return src;
    }

    auto op = output(Generic("Reshape", src));
    op->setDims(newDims);

    return op;
*/
//end of first implementation

/*
 //FIX ME fuse reshape
    //if(src->creatorLayer.lock()->type == "Reshape") // fuse reshapes
    if(src->getCreatorLayer().lock()->type == "Reshape") // fuse reshapes
    {
        src->setDims(newDims);
        return src;
    }
*/
 //latest implementation

    std::string name = "Reshape-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prms;
    prms.precision = g_layer_precision;
    prms.name = name;
    auto layer = std::make_shared<InferenceEngine::ReshapeLayer>(prms);
    layer->type = "Reshape";
    src >> layer;
    //addOutput(layer, src->getTensorDesc().getDims());

   /*
   brief A vector of sizes of the shape
   std::vector<int> shape;
   */

    layer->params["axis"] = std::to_string(layer->axis);
    layer->params["num_axes"] = std::to_string(layer->num_axes);

/*  //check if mandatory to provide shape
    for (int i = 0; i < newDims.size(); i++)
    layer->shape[i] = static_cast<int>(newDims[i]);
    // VectorToStringI(layer->shape)
    std::string result;
    const char sep = ',';
    for (auto it : layer->shape) {
        result += std::to_string(it) + sep;
    }
    if (!result.empty()) {
        result = result.substr(0, result.size() - 2);
    }

   layer->params["dim"] = result;
*/
   addOutput(layer, newDims);
   auto op = output(layer);
    //op->setDims(newDims);

/*
    //FIX ME : HACK for [VPU] Unsupported 1D dimensions
    if (op->getTensorDesc().getDims().size() == 1) {
    TensorDims dims = {1, newDims[0]};
    op->setDims(dims);
    #ifdef NNLOG
    ALOGI("Reshape oputput data set dims size = %lu ", op->getTensorDesc().getDims().size());
    #endif
    }
*/
    return op;

}

static OutputPort Softmax(const OutputPort &src)
{

    auto inputDims = src->getTensorDesc().getDims();
/*
    //handle 2D and 4D tensors
    TensorDims newDims;
    if (inputDims.size() == 2) {
        uint32_t batch_size = inputDims[0];//getSizeOfDimension(inputShape, 0);
        uint32_t input_size = sizeOf(inputDims) / batch_size; //getNumberOfElements(inputShape) / batch_size;

        newDims = {batch_size, input_size, 1, 1};
        inputDims = newDims;


    } else if (inputDims.size() == 4) {
        //dim = convertShapeToDims(inputShape);
        //newDims = inputDims;
    } else {
        #ifdef NNLOG
        ALOGI("Softmax only 2D and 4D tensors supported");
        #endif
        //return false;
    }

*/
    std::string name = "Softmax-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto l = std::make_shared<InferenceEngine::SoftMaxLayer>(prm);
    l->type = "SoftMax";
    src >> l;
    //addOutput(l, src->getTensorDesc().getDims());
    addOutput(l, inputDims);

    return output(l);
/*
    auto op = output(l);
    op->setDims(newDims);
    return op;
*/
}

inline OutputPort Gather(const std::vector<OutputPort> inputs, int axis = 1)
{
    std::string name = "Gather-"; // todo: make it unique
    name = name << layer_name_count++;
    InferenceEngine::LayerParams prm;
    prm.precision = g_layer_precision;
    prm.name = name;
    auto ret = std::make_shared<InferenceEngine::GenericLayer>(prm);
    ret->type = "Gather";
    addAttr(ret, "axis", axis);
    inputs[0] >> ret;
    inputs[1] >> ret;
    auto outDim = inputs[0]->getTensorDesc().getDims();
    //axis = static_cast<int>(outDim.size()) - axis - 1; // todo: we are all in reverse here :-(
    outDim[0] = inputs[1]->getTensorDesc().getDims()[1];
    addOutput(ret, outDim);
    return output(ret);
}


inline OutputPort operator+(const OutputPort &a, const OutputPort &b)
{
    return output(SumLayer::create(a, b));
}

inline OutputPort AddConst(IRDocument &doc, const OutputPort &src, const IRBlob::Ptr &biases) {
    // this depends on the plugin, see E-mail
    bool useScaleShift = false;

    if (useScaleShift) {
        return ScaleShiftNode(src, nullptr, biases);
    }
    // use const layer with elment wise add
    auto constNode = Generic("Const");
    doc.add(constNode);
    constNode->blobs["custom"] = biases;
    const auto constOut = addOutput(constNode, src->getTensorDesc().getDims());
    return src + constOut;
}

}  // namespace IRBuilder
