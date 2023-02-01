/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef UTILS_H
#define UTILS_H

#include <android-base/logging.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <android/log.h>
#include <hidlmemory/mapping.h>
#include <log/log.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fstream>
#include "Driver.h"
#include "IENetwork.h"
// May be move these out of utils??
#include "ie_blob.h"
#include "ie_common.h"

#undef LOG_TAG
#define LOG_TAG "Utils"

#if __ANDROID__
#include <hardware/hardware.h>
#endif

// unsigned int debugMask = ((1 << (L1 + 1)) - 1);

// extern unsigned int debugMask  = ((1 << (L1 + 1)) - 1);
using ::android::hidl::memory::V1_0::IMemory;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

enum DebugLevel {
    L0,
    L1,
    L2,
    L3,
    L4,
};

extern unsigned int debugMask;
// unsigned int debugMask = ((1 << (L1 + 1)) - 1);

enum PaddingScheme {
    kPaddingUnknown = 0,
    /**
     * SAME padding.
     * Padding on both ends are the "same":
     *     padding_to_beginning =  total_padding / 2
     *     padding_to_end       = (total_padding + 1)/2.
     * i.e., for even number of padding, padding to both ends are exactly
     * the same; for odd number of padding, padding to the ending is bigger
     * than the padding to the beginning by 1.
     *
     * total_padding is a function of input, stride and filter size.
     * It could be computed as follows:
     *    out_size = (input + stride - 1) / stride;
     *    needed_input = (out_size - 1) * stride + filter_size
     *    total_padding = max(0, needed_input - output_size)
     *  The computation is the same for the horizontal and vertical directions.
     */
    kPaddingSame = 1,
    /**
     * VALID padding.
     * No padding. When the input size is not evenly divisible by
     * the filter size, the input at the end that could not fill
     * the whole filter tile will simply be ignored.
     */
    kPaddingValid = 2,
};

#define VLOGDIMS(l, d, header)                                                       \
    do {                                                                             \
        auto size = (d).size();                                                      \
        ALOGV("%s: vectors {%d, %d, %d, %d}", header, (d)[0], size > 1 ? (d)[1] : 0, \
              size > 2 ? (d)[2] : 0, size > 3 ? (d)[3] : 0);                         \
    } while (0)

#define dumpOperand(index, model)                               \
    do {                                                        \
        const auto op = model.operands[index];                  \
        ALOGV("Operand (%zu) %s", index, toString(op).c_str()); \
    } while (0)

#define dumpOperation(operation)                             \
    do {                                                     \
        ALOGV("Operation: %s", toString(operation).c_str()); \
    } while (0)

#define WRONG_DIM (-1)

#undef nnAssert
#define nnAssert(v)                                                                            \
    do {                                                                                       \
        if (!(v)) {                                                                            \
            LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                       << "'\n";                                                               \
            abort();                                                                           \
        }                                                                                      \
    } while (0)

#define CHECK_OPERAND_2D(params, idx_x, idx_y)                                              \
    do {                                                                                    \
        ALOGI("As found in %s", __func__);                                                  \
        if (params.x < 0 || params.y < 0) {                                                 \
            ALOGI("Invalid Point2D Operands at index [%d ,%d] , aborting!!", idx_x, idx_y); \
            return false;                                                                   \
        }                                                                                   \
    } while (0)

#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16 0x7C00U

template <class T>
using vec = std::vector<T>;

typedef InferenceEngine::SizeVector TensorDims;
typedef InferenceEngine::Blob IRBlob;

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t zeroPoint;
    uint8_t* buffer;
    uint32_t length;
    OperandLifeTime lifetime;
    uint32_t numberOfUsesLeft;
    V1_2::Operand::ExtraParams extraParams;
    Shape shape() const {
        return {
            .type = type,
            .dimensions = dimensions,
            .scale = scale,
            .offset = zeroPoint,
        };
    }
};

// Used to keep a pointer to each of the memory pools.
struct RunTimePoolInfo {
    sp<IMemory> memory;
    hidl_memory hidlMemory;
    uint8_t* buffer;

    bool set(const hidl_memory& hidlMemory);
    bool update();
    bool unmap_mem();
};

template <typename T>
struct printHelper {
    static void print(const T&, const char*) {}
};

template <>
struct printHelper<int32_t> {
    static void print(const int32_t& value, const char* operand) {
        ALOGI("Operand: value: %d, %s", value, operand);
    }
};

template <>
struct printHelper<float> {
    static void print(const float& value, const char* operand) {
        ALOGI("Operand: value: %f, %s", value, operand);
    }
};

// small helper function to represent uint32_t value as float32
float asfloat(uint32_t v);

// Function to convert F32 into F16
float f16tof32(short x);

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
short f32tof16(float x);

void f16tof32Arrays(float* dst, const short* src, uint32_t& nelem, float scale = 1, float bias = 0);

void f32tof16Arrays(short* dst, const float* src, uint32_t& nelem, float scale = 1, float bias = 0);

TensorDims toDims(const vec<uint32_t>& dims);

TensorDims permuteDims(const TensorDims& src, const vec<unsigned int>& order);

// IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int> &order)
IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int>& order);

uint32_t getNumberOfElements(const vec<uint32_t>& dims);

size_t getSizeFromInts(int lower, int higher);

size_t sizeOfTensor(const TensorDims& dims);

// #ifdef NN_DEBUG
// template <typename T>
// void printBuffer(T* buf, int num, int items, const char* format, uint32_t buf_len) {
//     const size_t maxlen = 1024;
//     char str[maxlen] = {0};
//     uint32_t start = 0, n = 0;
//     while (n < num) {
//         int offset = 0;
//         n = (n + items) > num ? num : n + items;
//         offset = snprintf(str, sizeof(str) - strnlen(str, maxlen), "[%d->%d]:\t", start, n);
//         for (uint32_t i = start; i < n; i++) {
//             if (i < buf_len) {
//                 offset +=
//                     snprintf(str + offset, sizeof(str) - strnlen(str, maxlen), format, buf[i]);
//             }
//         }
//         start = n;
//         ALOGV("%s", str);
//     }
// }

// #endif

template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand) {
    const T* data = reinterpret_cast<const T*>(&model.operandValues[operand.location.offset]);
    return data[0];
}

int sizeOfData(OperandType type, std::vector<uint32_t> dims);

bool getGrpcSocketPath(char *socket_path);
bool getGrpcIpPort(char *ip_port);
template <typename T, typename S>
std::shared_ptr<T> As(const std::shared_ptr<S>& src) {
    return std::static_pointer_cast<T>(src);
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // UTILS_H
