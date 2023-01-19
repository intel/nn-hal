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
#include "utils.h"

#include <android-base/logging.h>
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <hidlmemory/mapping.h>
#include <log/log.h>
#include <sys/mman.h>

#include <sys/stat.h>
#include <vndk/hardware_buffer.h>
#include <fstream>

#undef LOG_TAG
#define LOG_TAG "Utils"

// inline unsigned int debugMask = ((1 << (L1 + 1)) - 1);

// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16 0x7C00U

using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

unsigned int debugMask = ((1 << (L1 + 1)) - 1);

unsigned short float2half(unsigned f) {
    unsigned f_exp, f_sig;
    unsigned short h_sgn, h_exp, h_sig;

    h_sgn = (unsigned short)((f & 0x80000000u) >> 16);
    f_exp = (f & 0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f & 0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                unsigned short ret = (unsigned short)(0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (unsigned short)(h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
            npy_set_floatstatus_overflow();
#endif
            return (unsigned short)(h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (f_exp < 0x33000000u) {
#if NPY_HALF_GENERATE_UNDERFLOW
            /* If f != 0, it underflowed to 0 */
            if ((f & 0x7fffffff) != 0) {
                npy_set_floatstatus_underflow();
            }
#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f & 0x007fffffu));
#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        if ((f_sig & (((unsigned)1 << (126 - f_exp)) - 1)) != 0) {
            npy_set_floatstatus_underflow();
        }
#endif
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if NPY_HALF_ROUND_TIES_TO_EVEN
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((f_sig & 0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
#else
        f_sig += 0x00001000u;
#endif
        h_sig = (unsigned short)(f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (unsigned short)(h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (unsigned short)((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f & 0x007fffffu);
#if NPY_HALF_ROUND_TIES_TO_EVEN
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig & 0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
#else
    f_sig += 0x00001000u;
#endif
    h_sig = (unsigned short)(f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if NPY_HALF_GENERATE_OVERFLOW
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
        npy_set_floatstatus_overflow();
    }
    return h_sgn + h_sig;
#else
    return h_sgn + h_exp + h_sig;
#endif
}

void floattofp16(short* dst, float* src, unsigned nelem) {
    unsigned i;
    unsigned short* _dst = (unsigned short*)dst;
    unsigned* _src = (unsigned*)src;

    for (i = 0; i < nelem; i++) _dst[i] = float2half(_src[i]);
}

// small helper function to represent uint32_t value as float32
float asfloat(uint32_t v) { return *reinterpret_cast<float*>(&v); }

// Function to convert F32 into F16
float f16tof32(short x) {
    // this is storage for output result
    uint32_t u = x;

    // get sign in 32bit format
    uint32_t s = ((u & 0x8000) << 16);

    // check for NAN and INF
    if ((u & EXP_MASK_F16) == EXP_MASK_F16) {
        // keep mantissa only
        u &= 0x03FF;

        // check if it is NAN and raise 10 bit to be align with intrin
        if (u) {
            u |= 0x0200;
        }

        u <<= (23 - 10);
        u |= EXP_MASK_F32;
        u |= s;
    } else if ((x & EXP_MASK_F16) ==
               0) {  // check for zero and denormals. both are converted to zero
        u = s;
    } else {
        // abs
        u = (u & 0x7FFF);

        // shift mantissa and exp from f16 to f32 position
        u <<= (23 - 10);

        // new bias for exp (f16 bias is 15 and f32 bias is 127)
        u += ((127 - 15) << 23);

        // add sign
        u |= s;
    }

    // finaly represent result as float and return
    return *reinterpret_cast<float*>(&u);
}

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
short f32tof16(float x) {
    // create minimal positive normal f16 value in f32 format
    // exp:-14,mantissa:0 -> 2^-14 * 1.0
    static float min16 = asfloat((127 - 14) << 23);

    // create maximal positive normal f16 value in f32 and f16 formats
    // exp:15,mantissa:11111 -> 2^15 * 1.(11111)
    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    // define and declare variable for intermidiate and output result
    // the union is used to simplify representation changing
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;

    // get sign in 16bit format
    uint32_t s = (v.u >> 16) & 0x8000;  // sign 16:  00000000 00000000 10000000 00000000

    // make it abs
    v.u &= 0x7FFFFFFF;  // abs mask: 01111111 11111111 11111111 11111111

    // check NAN and INF
    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return s | (v.u >> (23 - 10)) | 0x0200;  // return NAN f16
        } else {
            return s | (v.u >> (23 - 10));  // return INF f16
        }
    }

    // to make f32 round to nearest f16
    // create halfULP for f16 and add it to origin value
    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    // if input value is not fit normalized f16 then return 0
    // denormals are not covered by this code and just converted to 0
    if (v.f < min16 * 0.5F) {
        return s;
    }

    // if input value between min16/2 and min16 then return min16
    if (v.f < min16) {
        return s | (1 << 10);
    }

    // if input value more than maximal allowed value for f16
    // then return this maximal value
    if (v.f >= max16) {
        return max16f16 | s;
    }

    // change exp bias from 127 to 15
    v.u -= ((127 - 15) << 23);

    // round to f16
    v.u >>= (23 - 10);

    return v.u | s;
}

void f16tof32Arrays(float* dst, const short* src, uint32_t& nelem, float scale, float bias) {
    ALOGD("convert f16tof32Arrays...\n");
    const short* _src = reinterpret_cast<const short*>(src);

    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f16tof32(_src[i]) * scale + bias;
    }
}

void f32tof16Arrays(short* dst, const float* src, uint32_t& nelem, float scale, float bias) {
    ALOGD("convert f32tof16Arrays...");
    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f32tof16(src[i] * scale + bias);
    }
}

uint32_t getNumberOfElements(const vec<uint32_t>& dims) {
    uint32_t count = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        count *= dims[i];
    }
    return count;
}

size_t getSizeFromInts(int lower, int higher) {
    return (uint32_t)(lower) + ((uint64_t)(uint32_t)(higher) << 32);
}

TensorDims toDims(const vec<uint32_t>& dims) {
    TensorDims td;
    for (auto d : dims) td.push_back(d);
    return td;
}

template <typename T>
size_t product(const vec<T>& dims) {
    size_t rc = 1;
    for (auto d : dims) rc *= d;
    return rc;
}

TensorDims permuteDims(const TensorDims& src, const vec<unsigned int>& order) {
    TensorDims ret;
    for (size_t i = 0; i < src.size(); i++) {
        ret.push_back(src[order[i]]);
    }
    return ret;
}

// IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int> &order)
IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int>& order) {
    ALOGD("Permute");
    auto orig_dims = ptr->getTensorDesc().getDims();
    auto dims = permuteDims(orig_dims, order);
    ptr->getTensorDesc().setDims(dims);

    return ptr;
}

size_t sizeOfTensor(const TensorDims& dims) {
    size_t ret = dims[0];
    for (size_t i = 1; i < dims.size(); ++i) ret *= dims[i];
    return ret;
}

// TODO: short term, make share memory mapping and updating a utility function.
// TODO: long term, implement mmap_fd as a hidl IMemory service.
bool RunTimePoolInfo::set(const hidl_memory& hidlMemory) {
    this->hidlMemory = hidlMemory;
    buffer = nullptr;
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory = mapMemory(hidlMemory);
        if (memory == nullptr) {
            ALOGE("Can't map shared memory.");
            return false;
        }
        memory->update();
        buffer = reinterpret_cast<uint8_t*>(static_cast<void*>(memory->getPointer()));
        if (buffer == nullptr) {
            ALOGE("Can't access shared memory.");
            return false;
        }
        return true;
    } else if (memType == "mmap_fd") {
        size_t size = hidlMemory.size();
        int fd = hidlMemory.handle()->data[0];
        int prot = hidlMemory.handle()->data[1];
        size_t offset = getSizeFromInts(hidlMemory.handle()->data[2], hidlMemory.handle()->data[3]);
        buffer = static_cast<uint8_t*>(mmap(nullptr, size, prot, MAP_SHARED, fd, offset));
        if (buffer == MAP_FAILED) {
            ALOGE("Can't mmap the file descriptor.");
            return false;
        }
        return true;
    }
#if __ANDROID__
    else if (memType == "hardware_buffer_blob") {
        AHardwareBuffer* hardwareBuffer = nullptr;
        auto handle = hidlMemory.handle();
        auto format = AHARDWAREBUFFER_FORMAT_BLOB;
        auto usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        const uint32_t width = hidlMemory.size();
        const uint32_t height = 1;  // height is always 1 for BLOB mode AHardwareBuffer.
        const uint32_t layers = 1;  // layers is always 1 for BLOB mode AHardwareBuffer.
        const uint32_t stride = hidlMemory.size();

        AHardwareBuffer_Desc desc{
            .width = width,
            .format = format,
            .height = height,
            .layers = layers,
            .usage = usage,
            .stride = stride,
        };
        status_t status = AHardwareBuffer_createFromHandle(
            &desc, handle, AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_CLONE, &hardwareBuffer);
        if (status != NO_ERROR) {
            LOG(ERROR) << "RunTimePoolInfo Can't create AHardwareBuffer from handle. Error: "
                       << status;
            return false;
        }
        void* gBuffer = nullptr;
        status = AHardwareBuffer_lock(hardwareBuffer, usage, -1, nullptr, &gBuffer);
        if (status != NO_ERROR) {
            LOG(ERROR) << "RunTimePoolInfo Can't lock the AHardwareBuffer. Error: " << status;
            return false;
        }
        buffer = static_cast<uint8_t*>(gBuffer);
        return true;
    }
#endif
    else {
        ALOGE("unsupported hidl_memory type");
        return false;
    }
}

bool RunTimePoolInfo::unmap_mem() {
    if (buffer) {
        const size_t size = hidlMemory.size();
        if (hidlMemory.name() == "mmap_fd") {
            if (munmap(buffer, size)) {
                ALOGE("Unmap failed\n");
                return false;
            }
            buffer = nullptr;
        }
    }
    return true;
}

// Making sure the output data are correctly updated after execution.
bool RunTimePoolInfo::update() {
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory->commit();
        return true;
    } else if (memType == "mmap_fd") {
        int prot = hidlMemory.handle()->data[1];
        if (prot & PROT_WRITE) {
            size_t size = hidlMemory.size();
            return msync(buffer, size, MS_SYNC) == 0;
        }
    }
    // No-op for other types of memory.
    return true;
}

int sizeOfData(OperandType type, std::vector<uint32_t> dims) {
    int size;
    switch (type) {
        case OperandType::FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_INT32:
        case OperandType::INT32:
            size = 4;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::TENSOR_QUANT8_SYMM:
        case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
        case OperandType::TENSOR_BOOL8:
            size = 1;
            break;
        case OperandType::TENSOR_FLOAT16:
        case OperandType::TENSOR_QUANT16_SYMM:
        case OperandType::TENSOR_QUANT16_ASYMM:
        case OperandType::FLOAT16:
            size = 2;
            break;

        default:
            size = 0;
    }
    for (auto d : dims) size *= d;

    return size;
}

// TODO: Hide under debug flag or remove from code
bool createDirs(std::string path) {
    char delim = '/';
    int start = 0;

    auto pos = path.find(delim);
    while (pos != std::string::npos) {
        auto dir = path.substr(start, pos - start + 1);

        struct stat sb;
        if (!((stat(dir.c_str(), &sb) == 0) && (S_ISDIR(sb.st_mode)))) {
            if (mkdir(dir.c_str(), 0777) != 0) return false;
        }
        pos = path.find(delim, pos + 1);
    }

    return true;
}

bool getGrpcSocketPath(char *socket_path) {
    if (property_get("vendor.nn.hal.grpc_socket_path", socket_path, NULL) <= 0) {
        ALOGV("%s : failed to read vendor.nn.hal.grpc_socket_path", __func__);
        return false;
    }
    return true;
}

bool getGrpcIpPort(char *ip_port) {
    if (property_get("vendor.nn.hal.grpc_ip_port", ip_port, NULL) <= 0) {
        ALOGV("%s : failed to read vendor.nn.hal.grpc_ip_port", __func__);
        return false;
    }
    return true;
}

void writeBufferToFile(std::string filename, const float* buf, size_t length) {
    if (!createDirs(filename)) return;

    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
    for (size_t i = 0; i < length; i++) {
        ofs << buf[i] << "\n";
    }
    ofs.close();
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android