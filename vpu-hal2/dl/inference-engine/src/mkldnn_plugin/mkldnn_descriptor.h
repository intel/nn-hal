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
#pragma once

#include <memory>
#include <string>
#include <mkldnn.hpp>
#include <mkldnn/desc_iterator.hpp>

class MKLDNNDescriptor {
public:
    MKLDNNDescriptor() {}
    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::batch_normalization_forward::desc> desc);
    operator std::shared_ptr<mkldnn::batch_normalization_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_forward::desc> desc);
    operator std::shared_ptr<mkldnn::convolution_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_relu_forward::desc> desc);
    operator std::shared_ptr<mkldnn::convolution_relu_forward::desc>();

    MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_backward_data::desc> desc,
                     std::shared_ptr<mkldnn::convolution_forward::primitive_desc> prim);
    operator std::shared_ptr<mkldnn::convolution_backward_data::desc>();
    operator std::shared_ptr<mkldnn::convolution_forward::primitive_desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::inner_product_forward::desc> desc);
    operator std::shared_ptr<mkldnn::inner_product_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::lrn_forward::desc> desc);
    operator std::shared_ptr<mkldnn::lrn_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::pooling_forward::desc> desc);
    operator std::shared_ptr<mkldnn::pooling_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::relu_forward::desc> desc);
    operator std::shared_ptr<mkldnn::relu_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::roi_pooling_forward::desc> desc);
    operator std::shared_ptr<mkldnn::roi_pooling_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::softmax_forward::desc> desc);
    operator std::shared_ptr<mkldnn::softmax_forward::desc>();

    mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::engine &engine,
            const mkldnn::primitive_attr &attr = mkldnn::primitive_attr());
    mkldnn::primitive_desc_iterator * createPrimitiveDescriptorIteratorPtr(const mkldnn::engine &engine);

    size_t outputNumbers();
    size_t inputNumbers();

    operator bool();

private:
    class IDesc {
    public:
        virtual ~IDesc() {}
        virtual mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::engine &engine) = 0;
        virtual mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::primitive_attr &attr,
                                                                                  const mkldnn::engine &engine) = 0;
        virtual mkldnn::primitive_desc_iterator *createPrimitiveDescriptorIteratorPtr(const mkldnn::engine &engine) = 0;
    };

    template <class T>
    class DescFwdImpl: public IDesc {
        std::shared_ptr<T> desc;
    public:
        explicit DescFwdImpl(std::shared_ptr<T> d) : desc(d) {}

        mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::engine &engine) override {
            return mkldnn::primitive_desc_iterator(*desc, engine);
        }

        mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::primitive_attr &attr,
                                                                          const mkldnn::engine &engine) override {
            return mkldnn::primitive_desc_iterator(*desc, attr, engine);
        }

        mkldnn::primitive_desc_iterator *createPrimitiveDescriptorIteratorPtr(const mkldnn::engine &engine) override {
            return new mkldnn::primitive_desc_iterator(*desc, engine);
        }

        std::shared_ptr<T>& getPtr() {
            return desc;
        }
    };


    template <class T, class P>
    class DescBwdImpl: public IDesc {
        std::shared_ptr<T> desc;
        std::shared_ptr<P> prim;

    public:
        DescBwdImpl(std::shared_ptr<T> d, std::shared_ptr<P> p) : desc(d), prim(p) {}

        mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::engine &engine) override {
            return mkldnn::primitive_desc_iterator(*desc, engine, *prim);
        }

        mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::primitive_attr &attr,
                                                                          const mkldnn::engine &engine) override {
            return mkldnn::primitive_desc_iterator(*desc, attr, engine, *prim);
        }

        mkldnn::primitive_desc_iterator *createPrimitiveDescriptorIteratorPtr(const mkldnn::engine &engine) override {
            return new mkldnn::primitive_desc_iterator(*desc, engine, *prim);
        }

        std::shared_ptr<T>& getPtr() {
            return desc;
        }

        std::shared_ptr<P>& getPrimPtr() {
            return prim;
        }
    };

    std::shared_ptr<IDesc> desc;
};