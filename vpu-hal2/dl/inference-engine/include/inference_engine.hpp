// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief A header file that provides a set of convenience utility functions and the main include file for all other .h files.
 * @file inference_engine.hpp
 */
#pragma once

#include <vector>
#include <numeric>
#include <algorithm>

#include <ie_blob.h>
#include <ie_api.h>
#include <ie_error.hpp>
#include <ie_layers.h>
#include <ie_device.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_config.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include <cpp/ie_plugin_cpp.hpp>
#include <cpp/ie_executable_network.hpp>
#include <ie_version.hpp>

namespace InferenceEngine {
/**
 * @brief Gets the top n results from a tblob
 * @param n Top n count
 * @param input 1D tblob that contains probabilities
 * @param output Vector of indexes for the top n places
 */
template<class T>
inline void TopResults(unsigned int n, TBlob<T> &input, std::vector<unsigned> &output) {
    size_t input_rank = input.dims().size();
    if (!input_rank || !input.dims().at(input_rank - 1))
        THROW_IE_EXCEPTION << "Input blob has incorrect dimensions!";
    size_t batchSize = input.dims().at(input_rank - 1);
    std::vector<unsigned> indexes(input.size() / batchSize);

    n = static_cast<unsigned>(std::min<size_t>((size_t) n, input.size()));

    output.resize(n * batchSize);

    for (size_t i = 0; i < batchSize; i++) {
        size_t offset = i * (input.size() / batchSize);
        T *batchData = input.data();
        batchData += offset;

        std::iota(std::begin(indexes), std::end(indexes), 0);
        std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes),
                          [&batchData](unsigned l, unsigned r) {
                              return batchData[l] > batchData[r];
                          });
        for (unsigned j = 0; j < n; j++) {
            output.at(i * n + j) = indexes.at(j);
        }
    }
}

#ifdef AKS
#define TBLOB_TOP_RESULT(precision)\
    case InferenceEngine::Precision::precision  : {\
        using myBlobType = typename InferenceEngine::PrecisionTrait<Precision::precision>::value_type;\
        TBlob<myBlobType> &tblob = static_cast<TBlob<myBlobType> &>(input);\
        TopResults(n, tblob, output);\
        break;\
    }

#else
#define TBLOB_TOP_RESULT(precision)\
    case InferenceEngine::Precision::precision  : {\
        using myBlobType = typename InferenceEngine::PrecisionTrait<Precision::precision>::value_type;\
        TBlob<myBlobType> &tblob = dynamic_cast<TBlob<myBlobType> &>(input);\
        TopResults(n, tblob, output);\
        break;\
    }
#endif
/**
 * @brief Gets the top n results from a blob
 * @param n Top n count
 * @param input 1D blob that contains probabilities
 * @param output Vector of indexes for the top n places
 */
inline void TopResults(unsigned int n, Blob &input, std::vector<unsigned> &output) {
    switch (input.precision()) {
        TBLOB_TOP_RESULT(FP32);
        TBLOB_TOP_RESULT(FP16);
        TBLOB_TOP_RESULT(Q78);
        TBLOB_TOP_RESULT(I16);
        TBLOB_TOP_RESULT(U8);
        TBLOB_TOP_RESULT(I8);
        TBLOB_TOP_RESULT(U16);
        TBLOB_TOP_RESULT(I32);
        default:
            THROW_IE_EXCEPTION << "cannot locate blob for precision: " << input.precision();
    }
}

#undef TBLOB_TOP_RESULT

/**
 * @brief Copies a 8-bit RGB image to the blob.
 * Throws an exception in case of dimensions or input size mismatch
 * @tparam data_t Type of the target blob
 * @param RGB8 8-bit RGB image
 * @param RGB8_size Size of the image
 * @param blob Target blob to write image to
 */
template<typename data_t>
void copyFromRGB8(uint8_t *RGB8, size_t RGB8_size, InferenceEngine::TBlob<data_t> *blob) {
    if (4 != blob->dims().size())
        THROW_IE_EXCEPTION << "Cannot write data to input blob! Blob has incorrect dimensions size "
                           << blob->dims().size();
    auto num_channels = blob->dims()[2];  // because RGB
    auto num_images = blob->dims()[3];
    size_t w = blob->dims()[0];
    size_t h = blob->dims()[1];
    auto nPixels = w * h;

    if (RGB8_size != w * h * num_channels * num_images)
        THROW_IE_EXCEPTION << "input pixels mismatch, expecting " << w * h * num_channels
                           << " bytes, got: " << RGB8_size;

    std::vector<data_t *> dataArray;
    for (unsigned int n = 0; n < num_images; n++) {
        for (unsigned int i = 0; i < num_channels; i++) {
            if (!n && !i && dataArray.empty()) {
                dataArray.push_back(blob->data());
            } else {
                dataArray.push_back(dataArray.at(n * num_channels + i - 1) + nPixels);
            }
        }
    }
    for (unsigned int n = 0; n < num_images; n++) {
        for (unsigned int i = 0; i < nPixels; i++) {
            for (unsigned int j = 0; j < num_channels; j++) {
                dataArray.at(n * num_channels + j)[i] = RGB8[i * num_channels + j + n * num_channels * nPixels];
            }
        }
    }
    return;
}

/**
 * @brief Splits the RGB channels to either I16 Blob or float blob.
 * The image buffer is assumed to be packed with no support for strides.
 * @param imgBufRGB8 Packed 24bit RGB image (3 bytes per pixel: R-G-B)
 * @param lengthbytesSize Size in bytes of the RGB image. It is equal to amount of pixels times 3 (number of channels)
 * @param input Blob to contain the split image (to 3 channels)
 */
inline void ConvertImageToInput(unsigned char *imgBufRGB8, size_t lengthbytesSize, Blob &input) {
  #ifdef AKS
    if (input.precision() == Precision::ePrecision::FP32) {
    //TBlob<float> *float_input = dynamic_cast<TBlob<float> *>(&input);
    TBlob<float> *float_input = static_cast<TBlob<float> *>(&input);
    if (float_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, float_input);
    }

    if (input.precision() == Precision::ePrecision::FP16) {
    //TBlob<short> *short_input = dynamic_cast<TBlob<short> *>(&input);
    TBlob<short> *short_input = static_cast<TBlob<short> *>(&input);
    if (short_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, short_input);
    }
    if (input.precision() == Precision::ePrecision::U8 || input.precision() == Precision::ePrecision::I8) {
    //TBlob<uint8_t> *byte_input = dynamic_cast<TBlob<uint8_t> *>(&input);
    TBlob<uint8_t> *byte_input = static_cast<TBlob<uint8_t> *>(&input);
    if (byte_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, byte_input);
    }

  #else
    TBlob<float> *float_input = dynamic_cast<TBlob<float> *>(&input);
    if (float_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, float_input);

    TBlob<short> *short_input = dynamic_cast<TBlob<short> *>(&input);
    if (short_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, short_input);

    TBlob<uint8_t> *byte_input = dynamic_cast<TBlob<uint8_t> *>(&input);
    if (byte_input != nullptr) copyFromRGB8(imgBufRGB8, lengthbytesSize, byte_input);
#endif
}

/**
 * @brief Copies data from a certain precision to float
 * @param dst Pointer to an output float buffer, must be allocated before the call
 * @param src Source blob to take data from
 */
template<typename T>
void copyToFloat(float *dst, const InferenceEngine::Blob *src) {
    if (!dst) {
        return;
    }
    const InferenceEngine::TBlob<T> *t_blob = dynamic_cast<const InferenceEngine::TBlob<T> *>(src);
    if (t_blob == nullptr) {
       // THROW_IE_EXCEPTION << "input type is " << src->precision() << " but input is not " << typeid(T).name();  //aks fix me
    }

    const T *srcPtr = t_blob->readOnly();
    if (srcPtr == nullptr) {
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }
    for (size_t i = 0; i < t_blob->size(); i++) dst[i] = srcPtr[i];
}


}  // namespace InferenceEngine
