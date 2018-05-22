//
// INTEL CONFIDENTIAL
// Copyright 2016 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//
#include <details/ie_exception.hpp>
#include "mkldnn_descriptor.h"

mkldnn::primitive_desc_iterator MKLDNNDescriptor::createPrimitiveDescriptorIterator(const mkldnn::engine &engine,
        const mkldnn::primitive_attr &attr) {
    return desc->createPrimitiveDescriptorIterator(attr, engine);
}

mkldnn::primitive_desc_iterator *MKLDNNDescriptor::createPrimitiveDescriptorIteratorPtr(const mkldnn::engine &engine) {
    return desc->createPrimitiveDescriptorIteratorPtr(engine);
}

MKLDNNDescriptor::operator bool() {
    return desc.get() != nullptr;
}

size_t MKLDNNDescriptor::inputNumbers() {
    DescFwdImpl<mkldnn::roi_pooling_forward::desc> *roiPooling =
            dynamic_cast<DescFwdImpl<mkldnn::roi_pooling_forward::desc> *>(desc.get());
    if (roiPooling != nullptr) {
        return roiPooling->getPtr()->c_api_inputs.size();
    }
    return 1;
}

size_t MKLDNNDescriptor::outputNumbers() {
    return 1;
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::batch_normalization_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::batch_normalization_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::batch_normalization_forward::desc>() {
    DescFwdImpl<mkldnn::batch_normalization_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::batch_normalization_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::convolution_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_forward::desc>() {
    DescFwdImpl<mkldnn::convolution_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::convolution_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_relu_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::convolution_relu_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_relu_forward::desc>() {
    DescFwdImpl<mkldnn::convolution_relu_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::convolution_relu_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_backward_data::desc> desc,
                                   std::shared_ptr<mkldnn::convolution_forward::primitive_desc> prim) {
    this->desc.reset(
            new DescBwdImpl<mkldnn::convolution_backward_data::desc,
                    mkldnn::convolution_forward::primitive_desc>(desc, prim));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_backward_data::desc>() {
    DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc> *typeDesc =
            dynamic_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc,
                    mkldnn::convolution_forward::primitive_desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_forward::primitive_desc>() {
    DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc> *typeDesc =
            dynamic_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc,
                    mkldnn::convolution_forward::primitive_desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPrimPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::inner_product_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::inner_product_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::inner_product_forward::desc>() {
    DescFwdImpl<mkldnn::inner_product_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::inner_product_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::lrn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lrn_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::lrn_forward::desc>() {
    DescFwdImpl<mkldnn::lrn_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::lrn_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::pooling_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::pooling_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::pooling_forward::desc>() {
    DescFwdImpl<mkldnn::pooling_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::pooling_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::relu_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::relu_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::relu_forward::desc>() {
    DescFwdImpl<mkldnn::relu_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::relu_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::roi_pooling_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::roi_pooling_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::roi_pooling_forward::desc>() {
    DescFwdImpl<mkldnn::roi_pooling_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::roi_pooling_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::softmax_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::softmax_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::softmax_forward::desc>() {
    DescFwdImpl<mkldnn::softmax_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::softmax_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}
