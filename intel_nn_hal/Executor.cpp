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

#define LOG_TAG "Executor"


#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <thread>
#include <fstream>
#include "Executor.h"
#include <mutex>

#define DISABLE_ALL_QUANT


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace driver {
namespace executor {

//std::mutex g_num_mutex;

  #define NN_DEBUG

  enum DebugLevel {
      L0,
      L1,
      L2,
      L3,
      L4,
  };

  unsigned int debugMask = ((1 << (L1 + 1)) - 1);

  #ifdef NN_DEBUG
  #define VLOG(l, x, ...)                                                \
      do {                                                               \
          if (debugMask & (1 << l))                                      \
              ALOGI("[%s] " x, __FUNCTION__, ##__VA_ARGS__);             \
      } while(0)

  #define VLOGDIMS(l, d, header)                                         \
      do {                                                               \
          auto size = (d).size();                                        \
          VLOG(l, "%s: vectors {%d, %d, %d, %d}",                        \
                   header,(d)[0], size > 1 ? (d)[1] : 0,                 \
                  size > 2 ? (d)[2] : 0, size > 3 ? (d)[3] : 0);         \
      } while(0)

  #define dumpOperand(index)                                             	\
      do {                                                               	\
          const auto op = mModel->operands[index];                         \
          ALOGI("---------------------------------------------");         \
          ALOGI("Operand index: %d", index);                              \
          ALOGI("%s", toString(op).c_str());                              \
          ALOGI("---------------------------------------------");         \
      } while (0)

  #define dumpOperation(operation)                                        \
      do {                                                                \
          ALOGI("---------------------------------------------");         \
          ALOGI("Operation:");                                            \
          ALOGI("%s", toString(operation).c_str());                       \
          ALOGI("---------------------------------------------");         \
      } while (0)

  #define dumpOperationSupport(operation, support)                        \
      do {                                                                \
          ALOGI("---------------------------------------------");         \
          ALOGI("Operation support: %s", support ? "True":"False");       \
          ALOGI("%s", toString(operation).c_str());                       \
          ALOGI("---------------------------------------------");         \
      } while (0)

  #else
  #define VLOG(...)
  #define VLOGDIMS(l, d, header)
  #define dumpOperand(...)
  #define dumpOperation(operation)
  #define dumpOperationSupport(operation, support)
  #endif

  #define WRONG_DIM  (-1)

  #define nnAssert(v)                                                                            \
      do {                                                                                       \
          if (!(v)) {                                                                            \
              LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                         << "'\n";                                                               \
              abort();                                                                           \
          }                                                                                      \
      } while (0)


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

void calculateExplicitPadding(int32_t in_size, int32_t stride,
                              int32_t filter_size, int32_t padding_implicit,
                              int32_t* padding_head, int32_t* padding_tail) {
    *padding_head = 0;
    *padding_tail = 0;

    if (padding_implicit == kPaddingSame) {
        int32_t out_size = (in_size + stride - 1) / stride;
        int32_t tmp = (out_size - 1) * stride + filter_size;
        if (tmp > in_size) {
            *padding_head = (tmp - in_size) / 2;
            *padding_tail = (tmp - in_size) - *padding_head;
        }
    }
}

int32_t computeOutSize(int32_t imageSize, int32_t filterSize, int32_t stride,
                               int32_t paddingHead, int32_t paddingTail) {
    return (imageSize - filterSize + stride + paddingHead + paddingTail) / stride;
}

static inline size_t sizeOf(const TensorDims &dims)
{
    size_t ret = dims[0];
    for(int i = 1; i < dims.size(); ++i) ret *= dims[i];
    return ret;
}

//shape is nchw, dims depends on layout
TensorDims dimsToShape(const std::vector<uint32_t>& dims, Layout layout)
{

    VLOG(L3, "layout: %d", static_cast<int>(layout));
    VLOGDIMS(L3, dims, "dims");
    TensorDims shape;
    uint32_t n, c, h, w;
    //4-D
    switch (layout) {
        case NCHW:
        case OIHW:
            n = dims[0];
            c = dims[1];
            h = dims[2];
            w = dims[3];
            shape = {n, c, h, w};
            break;
        case NHWC:
            n = dims[0];
            h = dims[1];
            w = dims[2];
            c = dims[3];
            shape = {n, c, h, w};
            break;
        case IHWO:
            n = dims[3];
            c = dims[0];
            h = dims[1];
            w = dims[2];
            shape = {n, c, h, w};
            break;
        case C:
            n = dims[0];
            shape = {n};
            break;
        case NC:
            n = dims[0];
            c = dims[1];
            shape = {n, c};
            break;
        default:
            VLOG(L1, "unsupported layout %d", layout);
    }

    VLOGDIMS(L3, shape, "shape");
    return shape;
}

//shape is nchw, dims depends on format
std::vector<uint32_t>& shapeToDims(const TensorDims& shape, Layout layout)
{

    VLOG(L3, "layout: %d", static_cast<int>(layout));
    VLOGDIMS(L3, shape, "shape");
    uint32_t n, c, h, w;
    std::vector<uint32_t> dims;
    //1-D
    if (layout == C) {
            n = shape[0];
            dims = {n};
            return dims;
    }

    if (layout == NC) {
            n = shape[0];
            c = shape[1];
            dims = {n, c};
            return dims;
    }


    //4-D
    //vpu accept nchw or oihw.
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = shape[3];

    switch (layout) {
        case NCHW:
        case OIHW:
           dims = {n, c, h, w};
            break;
        case NHWC:
            dims = {n, h, w, c};
            break;
        case IHWO:
            dims = {c, h, w, n};
            break;
        default:
            VLOG(L1, "unsupported layout %d", layout);
    }

    VLOGDIMS(L3, dims, "dims");
    return dims;
}


unsigned short float2half(unsigned f)
{
    unsigned f_exp, f_sig;
    unsigned short h_sgn, h_exp, h_sig;

    h_sgn = (unsigned short) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                unsigned short ret = (unsigned short) (0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (unsigned short) (h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
            npy_set_floatstatus_overflow();
#endif
            return (unsigned short) (h_sgn + 0x7c00u);
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
            if ((f&0x7fffffff) != 0) {
                npy_set_floatstatus_underflow();
            }
#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));
#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        if ((f_sig&(((unsigned)1 << (126 - f_exp)) - 1)) != 0) {
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
        if ((f_sig&0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
#else
        f_sig += 0x00001000u;
#endif
        h_sig = (unsigned short) (f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (unsigned short) (h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (unsigned short) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f&0x007fffffu);
#if NPY_HALF_ROUND_TIES_TO_EVEN
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig&0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
#else
    f_sig += 0x00001000u;
#endif
    h_sig = (unsigned short) (f_sig >> 13);
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
void floattofp16(short *dst, float *src, unsigned nelem)
{
	unsigned i;
	unsigned short *_dst = (unsigned short *)dst;
	unsigned *_src = (unsigned *)src;

	for(i = 0; i < nelem; i++)
		_dst[i] = float2half(_src[i]);

}


// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 	 0x7F800000U
#define EXP_MASK_F16     0x7C00U


// small helper function to represent uint32_t value as float32
float asfloat(uint32_t v) {
    return *reinterpret_cast<float *>(&v);
}

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
    } else if ((x & EXP_MASK_F16) == 0) {  // check for zero and denormals. both are converted to zero
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
    return *reinterpret_cast<float *>(&u);
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

void f16tof32Arrays(float *dst, const short *src, uint32_t& nelem, float scale = 1, float bias = 0) {
    VLOG(L1, "convert f16tof32Arrays...\n");
    const short *_src = reinterpret_cast<const short *>(src);

    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f16tof32(_src[i]) * scale + bias;
    }
}

void f32tof16Arrays(short *dst, const float *src, uint32_t& nelem, float scale = 1, float bias = 0) {
    VLOG(L1, "convert f32tof16Arrays...");
    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f32tof16(src[i] * scale + bias);
        //VLOG(L1, "element no: %d", i);
    }
}

int sizeOfData(OperandType type, std::vector<uint32_t> dims)
{
    int size;
    switch(type) {
        case OperandType::FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_FLOAT32:
            size = 4;
            break;
        case OperandType::TENSOR_INT32:
            size = 4;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::INT32:
            size = 1;
            break;

        default:
            size = 0;

    }
    for (auto d : dims)
        size *= d;

    return size;
}

inline size_t getSizeFromInts(int lower, int higher)
{
    return (uint32_t)(lower) + ((uint64_t)(uint32_t)(higher) << 32);
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
  // TODO: Check buffer is at least as long as size of data.
  T* data = reinterpret_cast<T*>(info.buffer);
  return data[0];
}

// TODO: short term, make share memory mapping and updating a utility function.
// TODO: long term, implement mmap_fd as a hidl IMemory service.

bool RunTimePoolInfo::set(const hidl_memory& hidlMemory) {
    this->hidlMemory = hidlMemory;
    auto memType = hidlMemory.name();
    if (memType == "ashmem") {
        memory = mapMemory(hidlMemory);
        if (memory == nullptr) {
            LOG(ERROR) << "Can't map shared memory.";
            return false;
        }
        memory->update();
        buffer = reinterpret_cast<uint8_t*>(static_cast<void*>(memory->getPointer()));
        if (buffer == nullptr) {
            LOG(ERROR) << "Can't access shared memory.";
            return false;
        }
        return true;
    } else if (memType == "mmap_fd") {
        size_t size = hidlMemory.size();
        int fd = hidlMemory.handle()->data[0];
        int prot = hidlMemory.handle()->data[1];
        size_t offset = getSizeFromInts(hidlMemory.handle()->data[2],
                                        hidlMemory.handle()->data[3]);
        buffer = static_cast<uint8_t*>(mmap(nullptr, size, prot, MAP_SHARED, fd, offset));
        if (buffer == MAP_FAILED) {
            LOG(ERROR) << "Can't mmap the file descriptor.";
            return false;
        }
        return true;
    } else {
        LOG(ERROR) << "unsupported hidl_memory type";
        return false;
    }
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

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools) {
    poolInfos->resize(pools.size());
    for (size_t i = 0; i < pools.size(); i++) {
        auto& poolInfo = (*poolInfos)[i];
        if (!poolInfo.set(pools[i])) {
            LOG(ERROR) << "Could not map pool";
            return false;
        }
    }
    return true;
}


// Updates the RunTimeOperandInfo with the newly calculated shape.
// Allocate the buffer if we need to.
static bool setInfoAndAllocateIfNeeded(RunTimeOperandInfo* info, const Shape& shape) {
    // For user-provided model output operands, the parameters must match the Shape
    // calculated from the preparation step.
    if (info->lifetime == OperandLifeTime::MODEL_OUTPUT) {
        if (info->type != shape.type ||
            info->dimensions != shape.dimensions) {
            LOG(ERROR) << "Invalid type or dimensions for model output";
            return false;
        }
        if (info->type == OperandType::TENSOR_QUANT8_ASYMM &&
            (info->scale != shape.scale || info->zeroPoint != shape.offset)) {
            LOG(ERROR) << "Invalid scale or zeroPoint for model output";
            return false;
        }
    }
    info->type = shape.type;
    info->dimensions = shape.dimensions;
    info->scale = shape.scale;
    info->zeroPoint = shape.offset;
    if (info->lifetime == OperandLifeTime::TEMPORARY_VARIABLE && info->buffer == nullptr) {
        uint32_t length = sizeOfData(info->type, info->dimensions);
        info->buffer = new uint8_t[length];
        if (info->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

//OutputPort handleFusion(const Model &model, const Operation &op, OutputPort &out, const int fusionIndex);

/*
template <typename T>
TensorDims toDims(const vec<T> &dims)
{
    TensorDims td;
    for (auto d: dims) td.push_back(d);
    return td;
}
*/
uint32_t getNumberOfElements(const vec<uint32_t> &dims) {
    uint32_t count = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        count *= dims[i];
    }
    return count;
}

TensorDims toDims(const vec<uint32_t> &dims)
{
    TensorDims td;
    for (auto d: dims) td.push_back(d);
    return td;
}

template<typename T>
size_t product(const vec<T> &dims)
{
    size_t rc=1;
    for (auto d: dims) rc*= d;
    return rc;
}

TensorDims permuteDims(const TensorDims &src, const vec<unsigned int> &order)
{

  TensorDims ret;
	for (int i=0; i<src.size(); i++)
	{
		ret.push_back(src[order[i]]);
	}
	return ret;
}

//IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int> &order)
IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int> &order)
{
    VLOG(L1, "Permute");
    auto orig_dims = ptr->getTensorDesc().getDims();
    auto dims = permuteDims(orig_dims, order);
    ptr->getTensorDesc().setDims(dims);

    return ptr;
}

#define PARAM_I32(i) ParseOperationInput<int32_t>(mModel, operation, i)
#define PARAM_FP(i) ParseOperationInput<float>(mModel, operation, i)

template<typename T>
struct printHelper
{
   static void print(const T& value, const char* Obj) { }
};

template<>
struct printHelper<int32_t>
{
    static void print(const int32_t& value, const char* operand) {VLOG(L1, "Operand: value: %d, %s", value, operand); }
};

template<>
struct printHelper<float>
{
    static void print(const float& value, const char* operand) { VLOG(L1, "Operand: value: %f, %s", value, operand); }
};

template <typename T>
T Executor::ParseOperationInput(const Model* model, const Operation& operation, uint32_t index)
{
    uint32_t inputIndex = operation.inputs[index];
    const auto operand = mModel->operands[inputIndex];
    const auto value = GetConstOperand<T>(model, inputIndex);
    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
    VLOG(L1, "Operation input index: %d, operand index: %d", index, inputIndex);
    VLOG(L1, "Operation: %s", toString(operation).c_str());
    //VLOG(L1, "Operand: value: %d, %s", alue, toString(operand).c_str());
    printHelper<T>::print(value, toString(operand).c_str());
    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

    return value;
}

OutputPort Executor::handleFusion(const OutputPort &out, int32_t fusedOp)
{
    VLOG(L1, "fusedOp: %d", fusedOp);
    if (fusedOp == (int32_t)FusedActivationFunc::RELU) {
      VLOG(L1, "fusedOp is RELU");
      return ReLU(out);
    }
    else if (fusedOp == (int32_t)FusedActivationFunc::RELU1) {
      VLOG(L1, "fusedOp is RELU1");
      return Clamp(out, -1, 1);
    }
    else if (fusedOp == (int32_t)FusedActivationFunc::RELU6) {
      VLOG(L1, "fusedOp is RELU6");
      return Clamp(out, 0, 6);
    }

    VLOG(L1, "No ActivationFunc");
    return out;
}

template<typename T>
T Executor::GetConstFromBuffer(const uint8_t *buf, uint32_t len) {
    VLOG(L1, "buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        VLOG(L1, "fix me: typeid(T).name() should be %d bytes", sizeof(T));
        //fix me if buffer is of type float and if float and OperandLifeTime::CONSTANT_REFERENCE
        nnAssert(false);
    }
    return *(T *)(buf);
}


template<typename T>
std::vector<T> Executor::GetConstVecFromBuffer(const uint8_t *buf, uint32_t len) {
    int n = len/sizeof(T);
    if (n*sizeof(T) != len) {
        VLOG(L1, "typeid(T).name() should be  multiples of %d bytes", sizeof(T));
        nnAssert(false);
    }

    std::vector<T> ret;

    for (int i=0; i<n; i++)
    {
        ret.push_back(*(T*)buf);
        buf += sizeof(T);
    }

    return ret;
}

const uint8_t* Executor::GetOperandMemory(const Model *model, uint32_t index, uint32_t &len_out)
{
    //const auto op = model->operands[index];
    const auto op = mOperands[index];
    len_out = op.length;
    if (op.lifetime == OperandLifeTime::CONSTANT_COPY)
    {
      /*
        if (op.location.poolIndex != 0){
            ALOGE("CONSTANT_COPY expects poolIndex to be 0");
            nnAssert(false);
        //return &model.operandValues[op.location.offset];
      }*/
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_COPY");
        //return (const_cast<uint8_t*>(&model->operandValues[op.location.offset]));
        return op.buffer;

      //to.numberOfUsesLeft = 0;
    } else if (op.lifetime == OperandLifeTime::CONSTANT_REFERENCE)
    {
        //auto pool = model.pools[op.location.poolIndex];
        //return (pool + op.location.offset);
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_REFERENCE");
        //auto poolIndex = op.location.poolIndex;
        //nnAssert(poolIndex < mPoolInfos.size()); //aks fix me
        //auto& r = mPoolInfos[poolIndex];
        //return (const_cast<uint8_t*>(r.buffer + op.location.offset));
        return op.buffer;
    }
    else if (op.lifetime == OperandLifeTime::MODEL_INPUT
             || op.lifetime == OperandLifeTime::MODEL_OUTPUT
             || op.lifetime == OperandLifeTime::NO_VALUE)
    {
        //return const_cast<uint8_t*>(op.buffer);
        VLOG(L1, "operand lifetime OperandLifeTime::MODEL_INPUT||MODEL_OUTPUT||NO_VALUE");
        len_out = sizeOfData(op.type, op.dimensions);
        return nullptr;
    }
    else if (op.lifetime == OperandLifeTime::TEMPORARY_VARIABLE)
    {
        //return const_cast<uint8_t*>(op.buffer);
        VLOG(L1, "operand lifetime OperandLifeTime::TEMPORARY_VARIABLE");
        VLOG(L1, "operand is expected to be const, but lifetime is %d", op.lifetime);
        len_out = sizeOfData(op.type, op.dimensions);
        //nnAssert(false);
        return nullptr;
    }

    ALOGE("operand is expected to be const, but lifetime is %d", op.lifetime);
    nnAssert(false); //temp fix since some time const operand set as TEMPORARY_VARIABLE
    return nullptr;
}

template <typename T>
T Executor::GetConstOperand(const Model *model, uint32_t index)
{
    dumpOperand(index);
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(model, index, len);
    return GetConstFromBuffer<T>(buf, len);
}

template <typename T>
std::vector<T> Executor::GetConstVecOperand(const Model *model, uint32_t index)
{
    dumpOperand(index);
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(model, index, len);
    return GetConstVecFromBuffer<T>(buf, len);
}

//#define MYRIAD_FP32
IRBlob::Ptr Executor::GetConstWeightsOperandAsTensor(uint32_t index)
{
    return nullptr;
}

IRBlob::Ptr Executor::GetConstOperandAsTensor(uint32_t index)
{
    return nullptr;
}

Blob::Ptr Executor::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
    return nullptr;
}

OutputPort Executor::getPort(int index)
{
  VLOG(L1, "getPort\n");
	if (isConst(index))
	{
		VLOG(L1, "index is a const!");
		nnAssert(false);
	}
	//const auto op = mModel->operands[index];
  const auto op = mOperands[index];
	if (op.lifetime == OperandLifeTime::MODEL_INPUT)
	{
          VLOG(L1, "Model input operand\n");
          //std::ostringstream operandName; operandName << "operand."<<index;
          std::ostringstream operandName; operandName << "input";

          vec<unsigned int> order;
          if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
          else if (op.dimensions.size() == 2) order = {0, 1};
          else order = {0}; //(op.dimensions.size() < 2)

          auto operandInfo = mNet.createInput(operandName.str(), permuteDims(toDims(op.dimensions), order)); // NHWC -> NCHW

          mPorts[index] = operandInfo->getInputData(); //org

          //VLOG(L1, "getInputData\n");
          //mPorts[index]->setLayout(NHWC); // mPorts[i]->name
          //mPorts[index]->setPrecision(InferenceEngine::Precision::FP16);
          mPorts[index]->setPrecision(InferenceEngine::Precision::FP32);
          //TODO: workaround 3-D
          int dims_size = op.dimensions.size();

          VLOG(L1, "mPorts[%d] %s dims size %d", index, mPorts[index]->name.c_str(), dims_size);

          auto dims = permuteDims(toDims(op.dimensions), order);
          for (auto i = 0; i < dims.size(); i++)
          VLOG(L1, "input dims[%d] = %d & set input dims[%d] = %d ", i, op.dimensions[i], i, dims[i]);

          switch(dims_size) {
              case 2:
                  mPorts[index]->setLayout(NC);
                  break;
              case 4:
                  //mPorts[index]->setLayout(NHWC);
                  mPorts[index]->setLayout(NCHW);
                  break;
              case 1:
                  mPorts[index]->setLayout(C);
                  break;
              default:
                  VLOG(L1, "unsupported dims size %d", dims_size);
                  nnAssert(false);
          }

		      return mPorts[index];
	}
	if (op.lifetime == OperandLifeTime::MODEL_OUTPUT)
	{
		VLOG(L1, "Model output expected as input, not possible");
		nnAssert(false);
	}
	if (op.lifetime == OperandLifeTime::NO_VALUE)
	{
		VLOG(L1, "port is expected to be allocated for this as output from other layer");
		nnAssert(false);
	}
  if (op.lifetime == OperandLifeTime::TEMPORARY_VARIABLE)
  {
    VLOG(L1, "getport OperandLifeTime::TEMPORARY_VARIABLE\n");
    if (!mPorts[index]) nnAssert(false);
    VLOG(L1, "mPorts[%d] already allocated\n", index);
    return mPorts[index];
    //to.buffer = nullptr;
    //to.length = sizeOfData(to.type, to.dimensions);
    //nnAssert(true);
  }

	return nullptr;
}

//uint8_t* buffer;
// The length of the buffer.
//uint32_t length;

void Executor::SetOperandMemory(const Model *model, uint32_t index, uint32_t &len_out, const uint8_t *buf)
{

}



bool Executor::initializeRunTimeInfo(const std::vector<RunTimePoolInfo>& modelPoolInfos,
                                        const std::vector<RunTimePoolInfo>& requestPoolInfos) {
  //initialize runtime operand info from model.
  VLOG(L1, "initializeRunTimeInfo..");

  const size_t count = mModel->operands.size();
  mOperands.resize(count);
  mPorts.resize(count);

  // Start by setting the runtime info to what's in the model.
  for (size_t i = 0; i < count; i++) {
      const Operand& from = mModel->operands[i];
      RunTimeOperandInfo& to = mOperands[i];
      to.type = from.type;
      /*
      switch(from.type) {
          case OperandType::TENSOR_FLOAT32:
          case OperandType::FLOAT32:
              //nnAssert(to.scale == 0);
              to.type = OperandType::TENSOR_FLOAT32;
              VLOG(L1, "OperandType = %d\n", from.type);
              //port->setPrecision(InferenceEngine::Precision::FP32);

              break;
          case OperandType::INT32:
          case OperandType::UINT32:
              nnAssert(to.scale == 0);
          case OperandType::TENSOR_INT32:
              to.type = OperandType::TENSOR_INT32;
              //port->setPrecision(InferenceEngine::Precision::I32);
              VLOG(L1, "OperandType::TENSOR_INT32 and operand scale value = %.1f", to.scale);
              break;
          case OperandType::TENSOR_QUANT8_ASYMM:
              ALOGE("OperandType::TENSOR_QUANT8_ASYMM is not supported");
              nnAssert(to.scale != 0);
              //to.type = VpuDataType::U8;
              break;
          default:
              ALOGE("wrong operand type %d", from.type);;
              return false;
      }*/
      to.dimensions = from.dimensions;
      to.scale = from.scale;
      to.zeroPoint = from.zeroPoint;
      to.length = from.location.length;
      to.lifetime = from.lifetime;
      switch (from.lifetime) {
          case OperandLifeTime::TEMPORARY_VARIABLE:
              to.buffer = nullptr;
              to.numberOfUsesLeft = from.numberOfConsumers;
              break;
          case OperandLifeTime::CONSTANT_COPY:
              to.buffer = const_cast<uint8_t*>(&mModel->operandValues[from.location.offset]);
              to.numberOfUsesLeft = 0;
              break;
          case OperandLifeTime::CONSTANT_REFERENCE: {
              auto poolIndex = from.location.poolIndex;
              nnAssert(poolIndex < modelPoolInfos.size());
              auto& r = modelPoolInfos[poolIndex];
              to.buffer = r.buffer + from.location.offset;
              to.numberOfUsesLeft = 0;
              break;
          }
          case OperandLifeTime::MODEL_INPUT:
          case OperandLifeTime::MODEL_OUTPUT:
          case OperandLifeTime::NO_VALUE:
              to.buffer = nullptr;
              to.numberOfUsesLeft = 0;
              break;
          default:
              nnAssert(false);
              break;
      }
  }

  // Adjust the runtime info for the arguments passed to the model,
  // modifying the buffer location, and possibly the dimensions.

  auto updateForArguments = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                                const hidl_vec<RequestArgument>& arguments) {
      nnAssert(indexes.size() == arguments.size());
      for (size_t i = 0; i < indexes.size(); i++) {
          const uint32_t operandIndex = indexes[i];
          const RequestArgument& from = arguments[i];
          RunTimeOperandInfo& to = mOperands[operandIndex];
          if (from.dimensions.size() > 0) {
              // It's the responsibility of the caller to validate that
              // from.dimensions only modifies the dimensions that were
              // unspecified in the model.  That's the case in SampleDriver.cpp
              // with the call to validateRequest().
              // TODO make sure that's the case for the default CPU path.
              to.dimensions = from.dimensions;
          }
          if (from.hasNoValue) {
              to.lifetime = OperandLifeTime::NO_VALUE;
              nnAssert(to.buffer == nullptr);
          } else {
              auto poolIndex = from.location.poolIndex;
              nnAssert(poolIndex < requestPoolInfos.size());
              auto& r = requestPoolInfos[poolIndex];
              to.buffer = r.buffer + from.location.offset;
          }
      }
  };
  updateForArguments(mModel->inputIndexes, mRequest->inputs);
  updateForArguments(mModel->outputIndexes, mRequest->outputs);

  return true;

}


bool Executor::executeOperation(const Operation& operation){
    VLOG(L1, "get operation %d ready to add", operation.type);

    bool success = false;
    /*
    bool success = initializeRunTimeOperandInfo();


    if (!success) {
        VLOG(L1, "initializeRunTimeOperandInfo failed.");
        return false;
    }
    */
    dumpOperation(operation);
    switch (operation.type) {
        case OperationType::CONV_2D:
            success = operationConv2D(operation);
            break;
        case OperationType::DEPTHWISE_CONV_2D:
            success = operationDepthwiseConv2D(operation);
            break;
        case OperationType::MAX_POOL_2D:
            success = operationMaxPool2D(operation);
            break;
        case OperationType::AVERAGE_POOL_2D:
            success = operationAveragePool2D(operation);
            break;
        case OperationType::RELU:
            success = operationRELU(operation);
            break;
        case OperationType::RELU1:
            success = operationRELU1(operation);
            break;
        case OperationType::RELU6:
            success = operationRELU6(operation);
            break;
        case OperationType::LOGISTIC:
            success = operationLogisticSigmoid(operation);
            break;
        case OperationType::TANH:
            success = operationTANH(operation);
            break;
        case OperationType::CONCATENATION:
            success = operationConCat(operation);
            break;
        case OperationType::SOFTMAX:
            success = operationSoftmax(operation);
            break;
        case OperationType::LOCAL_RESPONSE_NORMALIZATION:
            success = operationLRN(operation);
            break;
        case OperationType::FULLY_CONNECTED:
            success = operationFullyConnected(operation);
            break;
        case OperationType::L2_NORMALIZATION:
            success = operationL2Normalization(operation);
            break;
        case OperationType::RESHAPE:
            success = operationReshape(operation);
            break;
        case OperationType::ADD:
            success = operationAdd(operation);
            break;
        default:
            VLOG(L1, "unsupported operation %d", operation.type);
            return false;
    }
    if (success == false) {
            VLOG(L1, "failed to convert operation %d", operation.type);
            return false;
    }
    VLOG(L1, "convert operation %d success", operation.type);


    //initializeInput();
    //finalizeOutput();

    //initialize IE operation input/output ports
    //    convertModel(mNet);

/*
    //debug graph
    mNet.buildNetwork();
    std::fstream dot;
    std::string graphfile("/data/local/graphfile");
    dot.open("/data/local/graph.dot", std::ios::out);
    mNet.save(graphfile);
    mNet.crateDotFile(dot);
    dot.close();

    VLOG(L1, "initialize ExecuteNetwork for device %s",
    InferenceEngine::TargetDeviceInfo::name(mTargetDevice));
    enginePtr = new ExecuteNetwork(mNet, mTargetDevice);
    enginePtr->loadNetwork();
*/
    return true;
}




void Executor::deinitialize()
{
    VLOG(L1, "deinitialize");
    delete enginePtr;
    enginePtr = nullptr;

    for (const auto& operand : mOperands) {
/*        for (const auto& buf : operand.buffer) {
            VLOG(L1, "free buffer %p of operand %p", buf, &operand);
            if (buf != nullptr)
            delete buf;
        }*/
        //VLOG(L1, "free buffer %p of operand %p", operand.buffer, &operand);
        //if (operand.buffer)
        //    delete operand.buffer;
    }
    VLOG(L1, "free engine");
}


#ifdef NN_DEBUG
template <typename T>
void printBuffer(int level, T* buf, int num, int items, const char* format)
{
    char str[1024];
    int start = 0;
    int n = 0;
    while (n < num) {
        int offset = 0;
        n = (n + items) > num ? num : n + items;
        offset = sprintf(str, "[%d->%d]:\t", start, n);
        for (int i = start; i < n; i++) {
            offset += sprintf(str + offset, format, buf[i]);
        }
        start = n;
        VLOG(level, "%s", str);
    }
}


void printOperandbuf(int level, const uint8_t* buffer, const std::vector<uint32_t>& dims, int limit = 0)
{
    auto dimsize = dims.size();
    auto type = OperandType::TENSOR_FLOAT32;//operand.type;
    int size = 1;
    for (int i = 0; i < dimsize; i++)
        size *= dims[i];

    if (limit > 0 && limit < size)
        size = limit;

    if (type == OperandType::TENSOR_FLOAT32) {
        //float *buf = static_cast<float *>(operand.buffer);
        printBuffer<float>(level, (float*)buffer, size, 10, "%f\t");
    } else if (type == OperandType::TENSOR_INT32) {
        //int32_t *buf = static_cast<int32_t *>(data_handle());
        //printBuffer<int32_t>(level, buf, size, 10, "%d\t");
    } else {
        VLOG(level, "Do not support type %d", type);
    }
}

#endif



int Executor::run(const Model& model, const Request& request,
        std::vector<RunTimePoolInfo>& modelPoolInfos,
        std::vector<RunTimePoolInfo>& requestPoolInfos)
{
    VLOG(L1, "run");

    mModel = &model;
    mRequest = &request; // TODO check if mRequest is needed

    initializeRunTimeInfo(modelPoolInfos, requestPoolInfos);

    // The model has serialized the operation in execution order.
    for (const auto& operation : mModel->operations) {
        bool n = executeOperation(operation);
        if (n) {
            //return n;
        }
    }

    auto inOutData = [this, &requestPoolInfos](const std::vector<uint32_t>& indexes,
                       const hidl_vec<RequestArgument>& arguments, bool inputFromRequest, ExecuteNetwork* enginePtr, std::vector<OutputPort> mPorts) {
        //do memcpy for input data
        for (size_t i = 0; i < indexes.size(); i++) {
            RunTimeOperandInfo& operand = mOperands[indexes[i]];
            const RequestArgument& arg = arguments[i];
            auto poolIndex = arg.location.poolIndex;
            nnAssert(poolIndex < requestPoolInfos.size());
            auto& r = requestPoolInfos[poolIndex];

            VLOG(L1, "Copy request input/output to model input/output");
            //std::ostringstream operandName; operandName << "operand."<<indexes[i]; //use mPort[i]->name
            if (inputFromRequest){
                //model/request oputput pointer pass to inference engine input
                //memcpy(operand.buffer, r.buffer + arg.location.offset, operand.length)

                auto inputBlob = GetInOutOperandAsBlob(operand, const_cast<uint8_t*>(r.buffer + arg.location.offset), operand.length); //if not doing memcpy
                VLOG(L1, "setBlob for mPorts[%d]->name %s", indexes[i], mPorts[indexes[i]]->name.c_str());
                enginePtr->setBlob(mPorts[indexes[i]]->name, inputBlob); //setInputBlob(const std::string &,IRBlob::Ptr);

              }
            else {
                //inference engine output pointer pass to model/request oputput
                //copy model oputput to request output
                //memcpy(r.buffer + arg.location.offset, operand.buffer, operand.length);

                auto outputBlob = GetInOutOperandAsBlob(operand, const_cast<uint8_t*>(r.buffer + arg.location.offset), operand.length); //if not doing memcpy
                enginePtr->setBlob(mPorts[indexes[i]]->name, outputBlob);

              }

        }

    };


    initializeInput();
    finalizeOutput();

    mNet.buildNetwork();
    std::fstream dot;
    std::string graphfile("/data/local/graphfile");
    dot.open("/data/local/graph.dot", std::ios::out);
    mNet.save(graphfile);
    mNet.crateDotFile(dot);
    dot.close();

    VLOG(L1, "initialize ExecuteNetwork for device %s",
    InferenceEngine::TargetDeviceInfo::name(mTargetDevice));
    enginePtr = new ExecuteNetwork(mNet, mTargetDevice);
    enginePtr->prepareInput();
    enginePtr->loadNetwork();

    VLOG(L1, "pass request inputs/outputs buffer to network/model respectively");

    inOutData(mModel->inputIndexes, request.inputs, true, enginePtr, mPorts);
    inOutData(mModel->outputIndexes, request.outputs, false, enginePtr, mPorts);

    VLOG(L1, "Run");

    //auto output = execute.Infer(input).wait();
    enginePtr->Infer();

//    VLOG(L1, "copy model output to request output");

    VLOG(L1, "update shared memories");

    for (auto& runtimeInfo : modelPoolInfos) {
        runtimeInfo.update();
    }
    for (auto& runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

#ifdef NN_DEBUG
    {
        VLOG(L1, "Model output0 are:");
        const RunTimeOperandInfo& output = mOperands[mModel->outputIndexes[0]];
        InferenceEngine::TBlob<float>::Ptr outBlob = enginePtr->getBlob(mPorts[mModel->outputIndexes[0]]->name);

        auto nelem = (outBlob->size() > 20 ? 20 : outBlob->size());
        for (int i = 0; i <  nelem; i++) {
        VLOG(L1, "outBlob elements %d = %f", i, outBlob->readOnly()[i]);
        }
        /*
        auto outbuf = outBlob->cbuffer();
        const uint8_t* buf1 = outbuf.as<uint8_t*>();
        memcpy(output.buffer, buf1, output.length);
        //uint8_t* buffer
        printOperandbuf(L2, const_cast<uint8_t*>(output.buffer), output.dimensions);
        */
        VLOG(L1, "Model input0 are:");
        const RunTimeOperandInfo& input = mOperands[mModel->inputIndexes[0]];
        InferenceEngine::TBlob<float>::Ptr inBlob = enginePtr->getBlob(mPorts[mModel->inputIndexes[0]]->name);
        nelem = (inBlob->size() > 20 ? 20 : inBlob->size());
        for (int i = 0; i < nelem ; i++) {
        VLOG(L1, "inBlob elements %d = %f", i, inBlob->readOnly()[i]);
        }

        /*
        auto inbuf = inBlob->cbuffer();
        uint8_t* buf2 = inbuf.as<uint8_t*>();
        memcpy(input.buffer, inbuf, input.length);
        printOperandbuf(L4, input.buffer, input.dimensions, 20);
        */

        /* //commented to run for MobileNet model and this only applicable for single cts opeation test
        for(const auto& op : mModel->operations) {
            const auto& o = mOperands[op.outputs[0]];
            InferenceEngine::TBlob<float>::Ptr opBlob = enginePtr->getBlob(mPorts[op.outputs[0]]->name);
            VLOG(L1, "Operation %d has output 0(lifetime %d) are:", op.type, o.lifetime);

            nelem = (opBlob->size() > 10 ? 10 : opBlob->size());
            for (int i = 0; i <  nelem; i++) {
            VLOG(L1, "operation output Blob elements %d = %f", i, opBlob->readOnly()[i]);
            }
            //printOperandbuf(L4, o.buffer, o.dimensions, 20);
        } */
    }
#endif


    mModel = nullptr;
    mRequest = nullptr;

    VLOG(L1, "Completed run normally");
    return 0;
}


template <typename T>
T getOperandConstVal(const Model* model, const Operand& operand)
{
    const T* data = reinterpret_cast<const T *>(&model->operandValues[operand.location.offset]);
    return data[0];
}

bool Executor::isConst(int index)
{
    VLOG(L1, "---------------------------------------------");
    VLOG(L1, "Operand index: %d", index);
    const auto op = mModel->operands[index];
    VLOG(L1, " %s", toString(op).c_str());
    bool ret = (op.lifetime == OperandLifeTime::CONSTANT_COPY || op.lifetime == OperandLifeTime::CONSTANT_REFERENCE);
    VLOG(L1, "%s", ret ? "Const" : "Non-Const");
    VLOG(L1, "---------------------------------------------");
    return ret;
}


bool Executor::operationAdd(const Operation& operation)
{
    VLOG(L1, "OperationType::ADD");
    OutputPort out;
    bool isIn0Const = isConst(operation.inputs[0]);
    bool isIn1Const = isConst(operation.inputs[1]);
    VLOG(L1, "isIn0Const = %d isIn1Const = %d \n", isIn0Const, isIn1Const);
    if (isIn0Const || isIn1Const) {
				if (isIn0Const && isIn1Const) {
					VLOG(L1, "adding 2 constants, we can do it now and put const as output");
					nnAssert(true);
				}
				// this will use ScaleShift
				if (isIn0Const) //if op.inputs[1] is a Model input
					out = AddConst(mNet, getPort(operation.inputs[1]),GetConstOperandAsTensor(operation.inputs[0]));
				else // isIn1Const is const //op.inputs[0] is a Model input
					out = AddConst(mNet, getPort(operation.inputs[0]),GetConstOperandAsTensor(operation.inputs[1]));
			} else { // both inputs[0] & inputs[1] are model inputs
				out = getPort(operation.inputs[0]) + getPort(operation.inputs[1]);
			}
    // check fusion
    VLOG(L1, "check fusion parameter = %d\n", PARAM_I32(2));

    mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(2));

    VLOG(L1, "add mPorts[%d]->name %s + mPorts[%d]->name %s  = mPorts[%d]->name %s \n", operation.inputs[0],
             isIn0Const ? "isIn0Const" : mPorts[operation.inputs[0]]->name.c_str(), operation.inputs[1],
                   isIn1Const ? "isIn1Const":  mPorts[operation.inputs[1]]->name.c_str(),
                      operation.outputs[0], mPorts[operation.outputs[0]]->name.c_str());

    return true;
}

bool Executor::operationAveragePool2D(const Operation& operation)
{
  VLOG(L1, "OperationType::AVERAGE_POOL_2D");
  /**
   * Performs a 2-D average pooling operation.
   *
   * The output dimensions are functions of the filter dimensions, stride, and
   * padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         sum_{i, j}(input[batch, row + i, col + j, channel]) / sum(1)
   *
   * Supported tensor {@link OperandType}:
   * * {@link OperandType::TENSOR_FLOAT32}
   * * {@link OperandType::TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" (i.e., Num_samples, Height, Width,
   * and Channels) data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
   *      the input.
   * * 1: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the left, in the ‘width’ dimension.
   * * 2: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the right, in the ‘width’ dimension.
   * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the top, in the ‘height’ dimension.
   * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the bottom, in the ‘height’ dimension.
   * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘width’ dimension.
   * * 6: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘height’ dimension.
   * * 7: An {@link OperandType::INT32} scalar, specifying the filter
   *      width.
   * * 8: An {@link OperandType::INT32} scalar, specifying the filter
   *      height.
   * * 9: An {@link OperandType::INT32} scalar, and has to be one of the
   *      {@link FusedActivationFunc} values. Specifies the activation to
   *      invoke on the result.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
   *      the input.
   * * 1: An {@link OperandType::INT32} scalar, specifying the implicit
   *      padding scheme, has to be one of the
   *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
   * * 2: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘width’ dimension.
   * * 3: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘height’ dimension.
   * * 4: An {@link OperandType::INT32} scalar, specifying the filter
   *      width.
   * * 5: An {@link OperandType::INT32} scalar, specifying the filter
   *      height.
   * * 6: An {@link OperandType::INT32} scalar, and has to be one of the
   *      {@link FusedActivationFunc} values. Specifies the activation to
   *      invoke on the result.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape
          [batches, out_height, out_width, depth].
   */

    auto input = getPort(operation.inputs[0]);
    const auto indims = input->getTensorDesc().getDims();

    Point2D pad_start;
    Point2D pad_end;
    Point2D stride;
    Point2D kernel;
    std::string padType;
    int  fusion_index = -1;

    if (operation.inputs.size() == 10) {
      padType = "explicit";
      pad_start = {PARAM_I32(1),PARAM_I32(3)};
      pad_end = {PARAM_I32(2), PARAM_I32(4)};
      stride = {PARAM_I32(5),PARAM_I32(6)};
      kernel = {PARAM_I32(7), PARAM_I32(8)};
      fusion_index = 9;
    } else if (operation.inputs.size() == 7) {//implicit padding
            const auto pad_type = PARAM_I32(1);
            int stride_width = PARAM_I32(2);
            int stride_height = PARAM_I32(3);
            int filter_width = PARAM_I32(4);
            int filter_height = PARAM_I32(5);
            fusion_index = 6;
            stride = {stride_width, stride_height};
            kernel = {filter_width, filter_height};

            int input_width = indims[3];
            int input_height = indims[2];

            int padding_left, padding_right;
            int padding_top, padding_bottom;

            if (pad_type == kPaddingSame) {
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

                calculateExplicitPadding(input_width, stride_width,
                                         filter_width, pad_type /*padding_implicit*/,
                                         &padding_left, &padding_right);
                calculateExplicitPadding(input_height, stride_height,
                                         filter_height, pad_type /*padding_implicit*/,
                                         &padding_top, &padding_bottom);

                pad_start = {padding_left, padding_top};
                pad_end = {padding_right, padding_bottom};
                padType = "same_upper";

            } else if (pad_type == kPaddingValid) {
                /**
                 * VALID padding.
                 * No padding. When the input size is not evenly divisible by
                 * the filter size, the input at the end that could not fill
                 * the whole filter tile will simply be ignored.
                 */
                pad_start = {0, 0};
                pad_end = {0, 0};
                padType = "valid";
            }

          }

      auto out = Pooling(input, kernel, stride, pad_start, pad_end, padType, InferenceEngine::PoolingLayer::PoolType::AVG);
      mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(fusion_index));

      return true;
}

bool Executor::operationMaxPool2D(const Operation& operation)
{
  VLOG(L1, "OperationType::MAX_POOL_2D");
    /*
     *  * Inputs:
     * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
     * 1: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
     * 2: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
     * 3: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
     * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
     * 5: An INT32 value, specifying the output stride in the ‘width’ dimension.
     * 6: An INT32 value, specifying the output stride in the ‘height’ dimension.
     * 7: An INT32 value, specifying the filter width.
     * 8: An INT32 value, specifying the filter height.
     * 9: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
     *    Specifies the activation to invoke on the result of each addition.
     */

     auto input = getPort(operation.inputs[0]);
     const auto indims = input->getTensorDesc().getDims();

     Point2D pad_start;
     Point2D pad_end;
     Point2D stride;
     Point2D kernel;
     std::string padType;
     int  fusion_index = -1;

     if (operation.inputs.size() == 10) {
       padType = "explicit";
       pad_start = {PARAM_I32(1),PARAM_I32(3)};
       pad_end = {PARAM_I32(2), PARAM_I32(4)};
       stride = {PARAM_I32(5),PARAM_I32(6)};
       kernel = {PARAM_I32(7), PARAM_I32(8)};
       fusion_index = 9;
       } else if (operation.inputs.size() == 7) {//PAD SAME
             const auto pad_type = PARAM_I32(1);
             int stride_width = PARAM_I32(2);
             int stride_height = PARAM_I32(3);
             int filter_width = PARAM_I32(4);
             int filter_height = PARAM_I32(5);
             fusion_index = 6;
             stride = {stride_width, stride_height};
             kernel = {filter_width, filter_height};

             int input_width = indims[3];
             int input_height = indims[2];

             int padding_left, padding_right;
             int padding_top, padding_bottom;

             if (pad_type == kPaddingSame) {
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

                 calculateExplicitPadding(input_width, stride_width,
                                          filter_width, pad_type /*padding_implicit*/,
                                          &padding_left, &padding_right);
                 calculateExplicitPadding(input_height, stride_height,
                                          filter_height, pad_type /*padding_implicit*/,
                                          &padding_top, &padding_bottom);

                 pad_start = {padding_left, padding_top};
                 pad_end = {padding_right, padding_bottom};
                 padType = "same_upper";

             } else if (pad_type == kPaddingValid) {
                 /**
                  * VALID padding.
                  * No padding. When the input size is not evenly divisible by
                  * the filter size, the input at the end that could not fill
                  * the whole filter tile will simply be ignored.
                  */
                 pad_start = {0, 0};
                 pad_end = {0, 0};
                 padType = "valid";
             }

           }

       auto out = Pooling(input, kernel, stride, pad_start, pad_end, padType, InferenceEngine::PoolingLayer::PoolType::MAX);
       mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(fusion_index));

       return true;
}

bool Executor::operationConCat(const Operation& operation)
{
  VLOG(L1, "OperationType::CONCATENATION");
  /*
   * Inputs:
   * 0 ~ n-1: The list on n input tensors, of shape [D0, D1, ..., Daxis(i), ..., Dm]
   * n: An INT32 value, specifying the concatenation axis.
   * n+1: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
   *    Specifies the activation to invoke on the result of each addition.
   */
  auto n = operation.inputs.size()-2;
  std::vector<OutputPort> inputs;
  for (int i=0; i<n; i++) inputs.push_back(getPort(operation.inputs[i]));
  auto out = Concat(inputs, PARAM_I32(n));
  mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(n+1));

  return true;
}

bool Executor::operationConv2D(const Operation& operation)
{
  VLOG(L1, "OperationType::CONV_2D");

      /**
       * Performs an 2-D convolution operation.
       *
       * The CONV_2D op sweeps a 2-D filter that can mix channels together over a
       * batch of images, applying the filter to each window of each image of the
       * appropriate size.
       *
       * The output dimensions are functions of the filter dimensions, stride, and
       * padding.
       *
       * The values in the output tensor are computed as:
       *
       *     output[batch, row, col, channel] =
       *         sum_{i, j} (
       *             input[batch, row + i, col + j, k] *
       *             filter[channel, row + i, col + j, k] +
       *             bias[channel]
       *         )
       *
       * Supported tensor {@link OperandType}:
       * * {@link OperandType::TENSOR_FLOAT32}
       * * {@link OperandType::TENSOR_QUANT8_ASYMM}
       *
       * Supported tensor rank: 4, with "NHWC" data layout.
       *
       * Both explicit padding and implicit padding are supported.
       *
       * Inputs (explicit padding):
       * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
       *      specifying the input.
       * * 1: A 4-D tensor, of shape
       *      [depth_out, filter_height, filter_width, depth_in], specifying the
       *      filter.
       * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
       *      For input tensor of {@link OperandType::TENSOR_FLOAT32}, the bias
       *      should also be of {@link OperandType::TENSOR_FLOAT32}. For input
       *      tensor of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias
       *      should be of {@link OperandType::TENSOR_INT32}, with zeroPoint of
       *      0 and bias_scale == input_scale * filter_scale.
       * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
       *      the left, in the ‘width’ dimension.
       * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
       *      the right, in the ‘width’ dimension.
       * * 5: An {@link OperandType::INT32} scalar, specifying the padding on
       *      the top, in the ‘height’ dimension.
       * * 6: An {@link OperandType::INT32} scalar, specifying the padding on
       *      the bottom, in the ‘height’ dimension.
       * * 7: An {@link OperandType::INT32} scalar, specifying the stride when
       *      walking through input in the ‘width’ dimension.
       * * 8: An {@link OperandType::INT32} scalar, specifying the stride when
       *      walking through input in the ‘height’ dimension.
       * * 9: An {@link OperandType::INT32} scalar, and has to be one of the
       *      {@link FusedActivationFunc} values. Specifies the activation to
       *      invoke on the result.
       *
       * Inputs (implicit padding):
       * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
       *      specifying the input.
       * * 1: A 4-D tensor, of shape
       *      [depth_out, filter_height, filter_width, depth_in], specifying the
       *      filter.
       * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
       *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
       *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
       *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
       *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
       *      bias_scale == input_scale * filter_scale.
       * * 3: An {@link OperandType::INT32} scalar, specifying the implicit
       *      padding scheme, has to be one of the
       *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
       * * 4: An {@link OperandType::INT32} scalar, specifying the stride when
       *      walking through input in the ‘width’ dimension.
       * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
      *       walking through input in the ‘height’ dimension.
       * * 6: An {@link OperandType::INT32} scalar, and has to be one of the
       *      {@link FusedActivationFunc} values. Specifies the activation to
       *      invoke on the result.
       *
       * Outputs:
       * * 0: The output 4-D tensor, of shape
       *      [batches, out_height, out_width, depth_out]. For output tensor of
       *      {@link OperandType::TENSOR_QUANT8_ASYMM}, the following condition
       *      must be satisfied: output_scale > input_scale * filter_scale.
       *
      CONV_2D = 3,

      ***/

  auto input = getPort(operation.inputs[0]);
  auto filter = GetConstOperandAsTensor(operation.inputs[1]); //OIHW
  auto bias = GetConstOperandAsTensor(operation.inputs[2]);

  const auto inputDims = input->getTensorDesc().getDims();
  const auto filterDims = filter->getTensorDesc().getDims();

  ConvolutionParams prms;
  //prms.weights = static_cast<IRBlob::Ptr>(filter); // permute OHWI to OIHW (0->0, 3->1, 1->2, 2->3)
  //const auto dims = prms.weights->getTensorDesc().getDims();
  //const auto indims = input->getTensorDesc().getDims();
  //int  fusion_index = -1;

  int batches = (int)inputDims[0];
  int in_channels = (int)inputDims[1];
  int input_height = (int)inputDims[2];
  int input_width = (int)inputDims[3];

  int filter_in = (int)filterDims[1];
  int filter_out = (int)filterDims[0];
  int filter_height = (int)filterDims[2];
  int filter_width = (int)filterDims[3];


  //int32_t padding_left, padding_right;
  //int32_t padding_top, padding_bottom;
  //int32_t stride_width, stride_height;

  uint32_t fusion_index = -1;

  if (operation.inputs.size() == 10) {
      prms.padType = "explicit";
      prms.pad_start = {PARAM_I32(3), PARAM_I32(5)};
      prms.pad_end = {PARAM_I32(4), PARAM_I32(6)};
      prms.stride = {PARAM_I32(7), PARAM_I32(8)};
      prms.kernel = {filter_width, filter_height};
      prms.num_output_planes = filter_out; //depth out
      fusion_index = 9;
  } else if (operation.inputs.size() == 7) {//PAD SAME
    const auto pad_type = PARAM_I32(3); //padding_implicit
    int stride_width = PARAM_I32(4);
    int stride_height = PARAM_I32(5);
    int padding_left, padding_right;
    int padding_top, padding_bottom;
/*
    int input_height = indims[2];
    int input_width = indims[3];

    int filter_height = dims[2];
    int filter_width = dims[3];
*/
    if (pad_type == kPaddingSame) {

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

        calculateExplicitPadding(input_width, stride_width,
                                 filter_width, pad_type /*padding_implicit*/,
                                 &padding_left, &padding_right);
        calculateExplicitPadding(input_height, stride_height,
                                 filter_height, pad_type /*padding_implicit*/,
                                 &padding_top, &padding_bottom);

        prms.pad_start = {padding_left, padding_top};
        prms.pad_end = {padding_right, padding_bottom};
        prms.padType = "same_upper";

    } else if (pad_type == kPaddingValid) {
          /**
           * VALID padding.
           * No padding. When the input size is not evenly divisible by
           * the filter size, the input at the end that could not fill
           * the whole filter tile will simply be ignored.
           */
          prms.pad_start = {0, 0};
          prms.pad_end = {0, 0};
          prms.padType = "valid";
      }
      prms.stride = {stride_width, stride_height};
      prms.kernel = {filter_width, filter_height};
      prms.num_output_planes = filter_out; //depth out
      fusion_index = 6;
   }

    if (bias->size() != prms.num_output_planes){
        VLOG(L1, "biases size mismatch filer's depth");
        nnAssert(false);
      }

    // input_size (validate)
    if (filter_in != in_channels){
        VLOG(L1, "filter depth_in size mismatch input depth");
        nnAssert(false);
      }


    //Reshape CONV_2D
/*
        //filter_in same as in_channels
        TensorDims newDims = {(uint32_t)filter_in, (uint32_t)filter_out, (uint32_t)filter_height, (uint32_t)filter_width}; //working for CTS
        //TensorDims newDims = {(uint32_t)filter_out, (uint32_t)filter_in, (uint32_t)filter_height, (uint32_t)filter_width};

        TensorDesc td(IRBuilder::g_layer_precision, newDims, {{newDims[2], newDims[3], newDims[0], newDims[1]}, {2, 3, 0, 1}}); //working for CTS


        //check diff combination btw TF lite and IE

        //using data_type = typename InferenceEngine::PrecisionTrait<IRBuilder::g_layer_precision>::value_type; //fix this to use calculate data type at runtime

        if (IRBuilder::g_layer_precision == InferenceEngine::Precision::FP32) {
          InferenceEngine::TBlob<float>::Ptr dst_blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
          dst_blob->allocate();

          for (size_t i = 0 ; i < filter->size(); i++) {
            dst_blob->buffer().as<float*>()[dst_blob->getTensorDesc().offset(i)] = filter->cbuffer().as<const float*>()[filter->getTensorDesc().offset(i)]; //set Layout::NCHW in td for src and dst
          }

          prms.weights = static_cast<IRBlob::Ptr>(dst_blob);
        }else {
          InferenceEngine::TBlob<short>::Ptr dst_blob = std::make_shared<InferenceEngine::TBlob<short>>(td); //short or uint16_t
          dst_blob->allocate();

          for (size_t i = 0 ; i < filter->size(); i++) {
            dst_blob->buffer().as<short*>()[dst_blob->getTensorDesc().offset(i)] = filter->cbuffer().as<const short*>()[filter->getTensorDesc().offset(i)]; //set Layout::NCHW in td for src and dst
          }

          prms.weights = static_cast<IRBlob::Ptr>(dst_blob);
        }
*/
    prms.weights = static_cast<IRBlob::Ptr>(filter); //org
    const auto weightsDims = prms.weights->getTensorDesc().getDims();

    //auto out = Convolution(input, prms) + bias;
    prms.biases = static_cast<IRBlob::Ptr>(bias);
    auto out = Convolution(input, prms);

    if (fusion_index < 0) {
        VLOG(L1, "invalid fusion index");
        nnAssert(false);
    }

    mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(fusion_index));

    VLOG(L1, "----------------------------------------------");
    VLOGDIMS(L1, inputDims, "inputs dims");
    VLOGDIMS(L1, filterDims, "filter dims");
    VLOGDIMS(L1, weightsDims, "weights dims");
    VLOG(L1, "----------------------------------------------");

    return true;
}

bool Executor::operationDepthwiseConv2D(const Operation& operation)
{
  VLOG(L1, "OperationType::DEPTHWISE_CONV_2D");
  /**
   * Performs a depthwise 2-D convolution operation.
   *
   * Given an input tensor of shape [batches, height, width, depth_in] and a
   * filter tensor of shape [1, filter_height, filter_width, depth_out]
   * containing depth_out convolutional filters of depth 1, DEPTHWISE_CONV
   * applies a different filter to each input channel (expanding from 1
   * channel to channel_multiplier channels for each), then concatenates the
   * results together.
   *
   * The output has depth_out = depth_in * depth_multiplier channels.
   * The output dimensions are functions of the filter dimensions, stride, and
   * padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[b, i, j, k * channel_multiplier + q] =
   *         sum_{di, dj} (
   *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
   *             filter[1, di, dj, k * channel_multiplier + q]
   *         )
   *
   * Supported tensor {@link OperandType}:
   * * {@link OperandType::TENSOR_FLOAT32}
   * * {@link OperandType::TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
   *      specifying the input.
   * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
   *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
   *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
   *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
   *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the left, in the ‘width’ dimension.
   * * 4: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the right, in the ‘width’ dimension.
   * * 5: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the top, in the ‘height’ dimension.
   * * 6: An {@link OperandType::INT32} scalar, specifying the padding on
   *      the bottom, in the ‘height’ dimension.
   * * 7: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘width’ dimension.
   * * 8: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘height’ dimension.
   * * 9: An {@link OperandType::INT32} scalar, specifying the depthwise
   *      multiplier.
   * * 10: An {@link OperandType::INT32} scalar, and has to be one of the
   *       {@link FusedActivationFunc} values. Specifies the activation to
   *       invoke on the result.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
   *      specifying the input.
   * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias. For input
   *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
   *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
   *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
   *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An {@link OperandType::INT32} scalar, specifying the implicit
   *      padding scheme, has to be one of the
   *      following values: {0 (NONE), 1 (SAME), 2 (VALID)}.
   * * 4: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘width’ dimension.
   * * 5: An {@link OperandType::INT32} scalar, specifying the stride when
   *      walking through input in the ‘height’ dimension.
   * * 6: An {@link OperandType::INT32} scalar, specifying the depthwise
   *      multiplier.
   * * 7: An {@link OperandType::INT32} scalar, and has to be one of the
   *      {@link FusedActivationFunc} values. Specifies the activation to
   *      invoke on the result.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape
   *      [batches, out_height, out_width, depth_out]. For output tensor of
   *      {@link OperandType::TENSOR_QUANT8_ASYMM}, the following condition
   *      must be satisfied: output_scale > input_scale * filter_scale.
   */


  auto input = getPort(operation.inputs[0]);
  //auto filter = GetConstOperandAsTensor(operation.inputs[1]); //NCHW [1, depth_out, filter_height, filter_width]
  auto filter = GetConstWeightsOperandAsTensor(operation.inputs[1]); //OIHW
  auto bias = GetConstOperandAsTensor(operation.inputs[2]);

  const auto inputDims = input->getTensorDesc().getDims();
  const auto filterDims = filter->getTensorDesc().getDims();

  ConvolutionParams prms;

  int batches = (int)inputDims[0];
  int in_channels = (int)inputDims[1];
  int input_height = (int)inputDims[2];
  int input_width = (int)inputDims[3];

  int filter_in = (int)filterDims[1];
  int filter_out = (int)filterDims[0];
  int filter_height = (int)filterDims[2];
  int filter_width = (int)filterDims[3];

  //int32_t padding_left, padding_right;
  //int32_t padding_top, padding_bottom;
  //int32_t stride_width, stride_height;

  int fusion_index = -1;
  int depth_multiplier = 0;


      if (operation.inputs.size() == 11) {
          prms.padType = "explicit";
          prms.pad_start = {PARAM_I32(3), PARAM_I32(5)};
          prms.pad_end = {PARAM_I32(4), PARAM_I32(6)};
          prms.stride = {PARAM_I32(7), PARAM_I32(8)};
          prms.kernel = {(int)filter_width, (int)filter_height};
          fusion_index = 10;
          prms.groups = in_channels; //working
          depth_multiplier = PARAM_I32(9);
          prms.num_output_planes = in_channels*depth_multiplier;//same as filter_out; //dims[0]; //depth out
      } else if (operation.inputs.size() == 8) {//PAD SAME
          const auto pad_type = PARAM_I32(3);
          int stride_width = PARAM_I32(4);
          int stride_height = PARAM_I32(5);

          int padding_left, padding_right;
          int padding_top, padding_bottom;

          if (pad_type == kPaddingSame) {
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
               calculateExplicitPadding(input_width, stride_width,
                                        filter_width, pad_type /*padding_implicit*/,
                                        &padding_left, &padding_right);
               calculateExplicitPadding(input_height, stride_height,
                                        filter_height, pad_type /*padding_implicit*/,
                                        &padding_top, &padding_bottom);

               prms.pad_start = {padding_left, padding_top};
               prms.pad_end = {padding_right, padding_bottom};
               prms.padType = "same_upper";

          } else if (pad_type == kPaddingValid) {
              /**
               * VALID padding.
               * No padding. When the input size is not evenly divisible by
               * the filter size, the input at the end that could not fill
               * the whole filter tile will simply be ignored.
               */
              prms.pad_start = {0, 0};
              prms.pad_end = {0, 0};
              prms.padType = "valid";
          }
          prms.stride = {stride_width, stride_height};
          prms.kernel = {(int)filter_width, (int)filter_height};
          fusion_index = 7;
          prms.groups = in_channels; //working
          depth_multiplier = PARAM_I32(6);
          prms.num_output_planes = in_channels*depth_multiplier;//same as filter_out; //dims[0]; //depth out
      }


  /*
  TF filter: 4-D with shape [filter_height, filter_width, in_channels, channel_multiplier].
  reshape to org layout of TF [filter_height, filter_width, in_channels, channel_multiplier]
  then permute 2, 3, 0, 1
  prms.weights = static_cast<IRBlob::Ptr>(Permute(filter, {2, 3, 0, 1}));

  group is same as in_channels and the number of output features map as in_channels*depth_multiplier and
  permute weights to (in_chennels, channel_multiplier, filter_height, filter_width) where assuming TF original input filter
  shape as [filter_height, filter_width,  in_channels, channel_multiplier] to preapare in the layout expected by IE
  */

/*
  //Reshape DEPTHWISE_CONV_2D
  //filter_out same as in_channels if depth_multiplier = 1
  TensorDims newDims = {(uint32_t)in_channels, (uint32_t)depth_multiplier, (uint32_t)filter_height, (uint32_t)filter_width}; //channel_multiplier == depth_multiplier working for CTS
  //TensorDims newDims = {(uint32_t)depth_multiplier, (uint32_t)in_channels, (uint32_t)filter_height, (uint32_t)filter_width};
  //TensorDims newDims = {1, in_channels*depth_multiplier, filter_height, filter_width}; //original filter shape //channel_multiplier == depth_multiplier

  //TensorDesc td(IRBuilder::g_layer_precision, newDims, {{newDims[2], newDims[3], newDims[0], newDims[1]}, {2, 3, 0, 1}}); //working for CTS
  TensorDesc td(IRBuilder::g_layer_precision, newDims, {{newDims[1], newDims[0], newDims[2], newDims[3]}, {1, 0, 2, 3}});
  //TensorDesc td(InferenceEngine::Precision::FP16, newDims, {{newDims[3], newDims[2], newDims[1], newDims[0]}, {3, 2, 1, 0}});

  //using data_type = typename InferenceEngine::PrecisionTrait<IRBuilder::g_layer_precision>::value_type;

  if (IRBuilder::g_layer_precision == InferenceEngine::Precision::FP32) {
    InferenceEngine::TBlob<float>::Ptr dst_blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
    dst_blob->allocate();

    for (size_t i = 0 ; i < filter->size(); i++) {
      dst_blob->buffer().as<float*>()[dst_blob->getTensorDesc().offset(i)] = filter->cbuffer().as<const float*>()[filter->getTensorDesc().offset(i)]; //set Layout::NCHW in td for src and dst
    }

    prms.weights = static_cast<IRBlob::Ptr>(dst_blob);
  }else {
    InferenceEngine::TBlob<short>::Ptr dst_blob = std::make_shared<InferenceEngine::TBlob<short>>(td); //short or uint16_t
    dst_blob->allocate();

    for (size_t i = 0 ; i < filter->size(); i++) {
      dst_blob->buffer().as<short*>()[dst_blob->getTensorDesc().offset(i)] = filter->cbuffer().as<const short*>()[filter->getTensorDesc().offset(i)]; //set Layout::NCHW in td for src and dst
    }

    prms.weights = static_cast<IRBlob::Ptr>(dst_blob);
  }

*/
  prms.weights = static_cast<IRBlob::Ptr>(filter);

  const auto weightDims = prms.weights->getTensorDesc().getDims();

  nnAssert(filter_out == in_channels * depth_multiplier);
  VLOG(L1, "batches %d, channels %d, input_height: %d, input_width %d",
           batches, in_channels, input_height, input_width);
  VLOG(L1, "filter_in %d, filter_out %d, filter_height: %d, filter_width %d",
           filter_in, filter_out, filter_height, filter_width);
  VLOG(L1, "depth multiplier %d", depth_multiplier);

  //auto out = Convolution(input, prms) + bias;
  prms.biases = static_cast<IRBlob::Ptr>(bias);
  auto out = Convolution(input, prms);

  if (fusion_index < 0) {
      VLOG(L1, "invalid fusion index");
      nnAssert(false);
  }

  mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(fusion_index));

  VLOG(L1, "----------------------------------------------");
  VLOGDIMS(L1, inputDims, "inputs dims");
  VLOGDIMS(L1, filterDims, "filter dims");
  VLOGDIMS(L1, weightDims, "weight dims");
  VLOG(L1, "----------------------------------------------");

  return true;

}

bool Executor::operationFullyConnected(const Operation& operation)
{
  VLOG(L1, "OperationType::FULLY_CONNECTED");
  /**
   * Denotes a fully (densely) connected layer, which connects all elements
   * in the input tensor with each element in the output tensor.
   *
   * This layer implements the operation:
   *
   *     outputs = activation(inputs * weights’ + bias)
   *
   * Supported tensor {@link OperandType}:
   * * {@link OperandType::TENSOR_FLOAT32}
   * * {@link OperandType::TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor of at least rank 2, specifying the input. If rank is
   *      greater than 2, then it gets flattened to a 2-D Tensor. The
   *      (flattened) 2-D Tensor is reshaped (if necessary) to
   *      [batch_size, input_size], where "input_size" corresponds to the
   *      number of inputs to the layer, matching the second dimension of
   *      weights, and "batch_size" is calculated by dividing the number of
   *      elements by "input_size".
   * * 1: A 2-D tensor, specifying the weights, of shape
   *      [num_units, input_size], where "num_units" corresponds to the number
   *      of output nodes.
   * * 2: A 1-D tensor, of shape [num_units], specifying the bias. For input
   *      tensor of {@link OperandType::TENSOR_FLOAT32}, the bias should
   *      also be of {@link OperandType::TENSOR_FLOAT32}. For input tensor
   *      of {@link OperandType::TENSOR_QUANT8_ASYMM}, the bias should be
   *      of {@link OperandType::TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An {@link OperandType::INT32} scalar, and has to be one of the
   *      {@link FusedActivationFunc} values. Specifies the activation to
   *      invoke on the result.
   *
   * Outputs:
   * * 0: The output tensor, of shape [batch_size, num_units]. For output
   *      tensor of {@link OperandType::TENSOR_QUANT8_ASYMM}, the following
   *      condition must be satisfied:
   *      output_scale > input_scale * filter_scale.

  FULLY_CONNECTED = 9,
   */

    auto input = getPort(operation.inputs[0]);
    auto weights = GetConstOperandAsTensor(operation.inputs[1]);
    auto bias = GetConstOperandAsTensor(operation.inputs[2]);

    auto inputDims = input->getTensorDesc().getDims();
    for (auto i = 0; i < inputDims.size(); i++)
    VLOG(L1, "input dims[%d] = %d ", i, inputDims[i]);

    auto weightsDims = weights->getTensorDesc().getDims();
    for (auto i = 0; i < weightsDims.size(); i++)
    VLOG(L1, "weights dims[%d] = %d ", i, weightsDims[i]);

    auto biasDims = bias->getTensorDesc().getDims();

    //input is [batch_size, input_size], weights is [num_unit, input_size]
    //nnAssert(inputDims[1] == weightsDims[1]);

    nnAssert(inputDims.size() >= 2);
    nnAssert(weightsDims.size() == 2);
    uint32_t numInputElements = sizeOf(inputDims);
    uint32_t num_units  = weightsDims[0];
    uint32_t input_size = weightsDims[1];
    uint32_t batch_size = numInputElements / input_size;
    nnAssert(biasDims[0] == num_units);
    nnAssert(input_size * batch_size == numInputElements);

    if (inputDims.size()>2)
    {
        // todo: could be we need to rotate the input weights to reflect the different layout of input tensor
        // when it is not 2D: NHWC vs NCHW in IE
        //Reshape
        //input = Reshape({inputDims[0], product(inputDims)/inputDims[0]}, input);

        TensorDims outDims = {(uint32_t)-1, numInputElements/batch_size};  //fix me: find correct outDims and if -1 is fine

        int strechDim = -1;
        auto numOutputElements = 1; //shape
        for (auto i = 0; i < outDims.size(); i++) {
            VLOG(L1, "shape of output tensor outDims[%d] = %d ", i, outDims[i]);
            if ((int)outDims[i] < 0) {
                strechDim = i; //strechdim
                VLOG(L1, "strechDim = %d", i);
                continue;
            }
            numOutputElements *= outDims[i]; //shape
        }
        if (strechDim >= 0) {
            auto strechValue = numInputElements / numOutputElements;
            outDims[strechDim] = (uint32_t) strechValue;
            numOutputElements *= strechValue;

            VLOG(L1, "numInputElements = %d, index = %d, outDims[index] = %d", numInputElements, strechDim, outDims[strechDim]);
        }

        input = Reshape(outDims, input);

/*
        //Reshape
        TensorDims newDims = {batch_size, input_n_elements/batch_size};

        auto precision = input->getPrecision();
        if (precision == InferenceEngine::Precision::FP16) {
          TensorDesc td(InferenceEngine::Precision::FP16, newDims, {{newDims[0], newDims[1]}, {0, 1}});

          InferenceEngine::TBlob<short>::Ptr dst_input_blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
          dst_input_blob->allocate();

          if (input->cbuffer() != nullptr) {
              for (size_t i = 0 ; i < input->size(); i++) {
                dst_input_blob->buffer().as<short*>()[dst_blob->getTensorDesc().offset(i)] = input->cbuffer().as<const short*>()[filter->getTensorDesc().offset(i)]; //set Layout::NCHW in td for src and dst
              }
          }
        }
        else if (precision == InferenceEngine::Precision::FP32) {
          TensorDesc td(InferenceEngine::Precision::FP32, newDims, {{newDims[0], newDims[1]}, {0, 1}});

          InferenceEngine::TBlob<float>::Ptr dst_input_blob = std::make_shared<InferenceEngine::TBlob<float>>(td);

          if (input->cbuffer() != nullptr){
              dst_input_blob->allocate();
              for (size_t i = 0 ; i < input->size(); i++) {
                dst_input_blob->buffer().as<float*>()[dst_blob->getTensorDesc().offset(i)] = input->cbuffer().as<const float*>()[filter->getTensorDesc().offset(i)]; //set Layout::NCHW in td for src and dst
              }
          }
        }

        input = static_cast<IRBlob::Ptr>(dst_input_blob);
*/
    }

    const auto newInputDims = input->getTensorDesc().getDims();

/*
    //FIX ME : Work around since input size indims[0] != output nodes (wdims[0])
    auto dims = permuteDims(weights->getTensorDesc().getDims(), {0, 1});
    dims[0] = indims[0];
    weights->getTensorDesc().setDims(dims);
    //WA end
*/
    auto out = weights*input + bias;


    mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(3));

    VLOG(L1, "----------------------------------------------");
    VLOGDIMS(L1, inputDims, "inputs dims");
    VLOGDIMS(L1, newInputDims, "newInput dims");
    VLOGDIMS(L1, weightsDims, "weights dims");
    VLOG(L1, "----------------------------------------------");

    return true;
}

bool Executor::operationL2Normalization(const Operation& operation)
{
  VLOG(L1, "OperationType::L2_NORMALIZATION");
  /*
  * Inputs:
  * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
  *
  * Ouputs:
  * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
  */
  //mPorts[operation.outputs[0]] = L2Normalization(getPort(operation.inputs[0]), true, false);
  mPorts[operation.outputs[0]] = L2Normalization(getPort(operation.inputs[0]), false, false); //passing accross false
  return true;
}

bool Executor::operationLRN(const Operation& operation)
{
  VLOG(L1, "OperationType::LOCAL_RESPONSE_NORMALIZATION");

  /*
   * * Inputs:
   * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * 1: An INT32 value, specifying the radius of the normalization window.
   * 2: A FLOAT32 value, specifying the bias, must not be zero.
   * 3: A FLOAT32 value, specifying the scale factor, alpha.
   * 4: A FLOAT32 value, specifying the exponent, beta.
   */

  float alpha = PARAM_FP(3);
  float beta = PARAM_FP(4);
  int size = PARAM_I32(1);
  float k = PARAM_FP(2);
  //mPorts[operation.outputs[0]] = LRN(getPort(operation.inputs[0]), alpha, beta, size, true, k);
  mPorts[operation.outputs[0]] = LRN(getPort(operation.inputs[0]), alpha, beta, size, false, k);

  return true;
}

bool Executor::operationLogisticSigmoid(const Operation& operation)
{
  VLOG(L1, "OperationType::LOGISTIC");
  mPorts[operation.outputs[0]] = Sigmoid(getPort(operation.inputs[0]));

  return true;
}
/*
bool Executor::operationLSTM(const Operation& operation)
{
    VLOG("operation type LSTM is supported, but not yet in this implementation");
    nnAssert(true);

    //return true;
}
*/
bool Executor::operationMUL(const Operation& operation)
{
    mPorts[operation.outputs[0]] = handleFusion(getPort(operation.inputs[0])*getPort(operation.inputs[1]), PARAM_I32(2));
    return true;
}

bool Executor::operationRELU(const Operation& operation)
{
    VLOG(L1, "OperationType::RELU");
    mPorts[operation.outputs[0]] = ReLU(getPort(operation.inputs[0]));
    return true;
}

bool Executor::operationRELU1(const Operation& operation)
{
    VLOG(L1, "OperationType::RELU1");
    /**
     * Computes rectified linear 1 activation on the input tensor element-wise.
     *
     * In details:
     *     output = min(1.f, max(-1.f, input))
     *
     * Supported tensor types: {@link OperandType::TENSOR_FLOAT32}
     *                         {@link OperandType::TENSOR_QUANT8_ASYMM}
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * 0: A tensor, specifying the input.
     *
     * Ouputs:
     * 0: The output tensor of same shape as input0.
     */

    mPorts[operation.outputs[0]] = Clamp(getPort(operation.inputs[0]),-1,1);
    return true;
}

bool Executor::operationRELU6(const Operation& operation)
{
    VLOG(L1, "OperationType::RELU6");
    /**
     * Computes rectified linear 6 activation on the input tensor element-wise.
     *
     * In details:
     *     output = min(6, max(0, input))
     *
     * Supported tensor types: {@link OperandType::TENSOR_FLOAT32}
     *                         {@link OperandType::TENSOR_QUANT8_ASYMM}
     * Supported tensor rank: up to 4.
     *
     * Inputs:
     * 0: A tensor, specifying the input.
     *
     * Ouputs:
     * 0: The output tensor of same shape as input0.
     */

    mPorts[operation.outputs[0]] = Clamp(getPort(operation.inputs[0]),0,6);
    return true;
}

bool Executor::operationReshape(const Operation& operation)
{
    VLOG(L1, "OperationType::RESHAPE");

     /**
      * Reshapes a tensor.
      *
      * Given tensor, this operation returns a tensor that has the same values as tensor,
      * but with a newly specified shape.
      *
      * Supported tensor types: {@link OperandType::TENSOR_FLOAT32}
      *                         {@link OperandType::TENSOR_QUANT8_ASYMM}
      * Supported tensor rank: up to 4.
      *
      * Inputs:
      * 0: A tensor, specifying the tensor to be reshaped.
      * 1: A 1-D tensor of type {@link OperandType::TENSOR_INT32}, defining the shape
      *    of the output tensor. The number of elements implied by shape must be the same
      *    as the number of elements in the input tensor.
      *
      * Ouputs:
      * 0: The output tensor, of shape specified by the input shape.
      */
     /* note: todo: We need to be careful here, inter-tensors are in different order,
      *       could be we need to reflect this also in reshape..
      */

    auto input = getPort(operation.inputs[0]);
    auto inDims = input->getTensorDesc().getDims();

    auto outDims = toDims(GetConstVecOperand<uint32_t>(mModel, operation.inputs[1]));

    // Reshape allows one of the targetDims components to have the
    // special -1 value, meaning it will be calculated automatically based on the
    // input. Here we calculate what that dimension should be so that the number
    // of output elements in the same as the number of input elements.

    //auto numInputElements = getNumberOfElements(static_cast<const vec<uint32_t>> (inDims));
    auto numInputElements = sizeOf(inDims);  //getNumberOfElements

    int strechDim = -1;
    auto numOutputElements = 1; //shape
    for (auto i = 0; i < outDims.size(); i++) {
        VLOG(L1, "operand1: shape of output tensor outDims[%d] = %d ", i, outDims[i]);
        if ((int)outDims[i] < 0) {
            strechDim = i; //strechdim
            VLOG(L1, "strechDim = %d", i);
            continue;
        }
        numOutputElements *= outDims[i]; //shape
    }
    if (strechDim >= 0) {
        auto strechValue = numInputElements / numOutputElements;
        outDims[strechDim] = (uint32_t) strechValue;
        numOutputElements *= strechValue;

        VLOG(L1, "numInputElements or size = %d, index = %d, outDims[index] = %d", numInputElements, strechDim, outDims[strechDim]);
    }

    for (auto i = 0; i < outDims.size(); i++)
        VLOG(L1, "operand1: shape of output tensor outDims[%d] = %d ", i, outDims[i]);
    if (numInputElements != numOutputElements){
      VLOG(L1, "numInputElements is not equal to numOutputElements", numInputElements, numOutputElements);
      nnAssert(false);
    }
//Note: " error [VPU] Unsupported 1 D dimensions" for reshape output and fix me
    mPorts[operation.outputs[0]] = Reshape(outDims, input);

    return true;

}

bool Executor::operationSoftmax(const Operation& operation)
{
    VLOG(L1, "OperationType::SOFTMAX");

    /**
     * Computes the softmax activation on the input tensor element-wise, per
     * batch, by normalizing the input vector so the maximum coefficient is
     * zero.
     *
     * The output is calculated using this formula:
     *
     *     output[batch, i] =
     *         exp((input[batch, i] - max(input[batch, :])) * beta) /
     *         sum_{k}{exp((input[batch, k] - max(input[batch, :])) * beta)}
     *
     * Supported tensor {@link OperandType}:
     * * {@link OperandType::TENSOR_FLOAT32}
     * * {@link OperandType::TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 2 or 4.
     *
     * Inputs:
     * * 0: A 2-D or 4-D tensor, specifying the tensor to be reshaped.
     * * 1: An {@link OperandType::FLOAT32} scalar, specifying the positive
     *      scaling factor for the exponent, beta.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *      For {@link OperandType::TENSOR_QUANT8_ASYMM},
     *      the scale must be 1.f / 256 and the zeroPoint must be 0.
     */

    auto input = getPort(operation.inputs[0]);

/*
    //handle 2D and 4D tensors
    auto inputDims = input->getTensorDesc().getDims();

    if (inputDims.size() == 2) {
        uint32_t batch_size = inputDims[0];//getSizeOfDimension(inputShape, 0);
        uint32_t input_size = sizeOf(inputDims) / batch_size; //getNumberOfElements(inputShape) / batch_size;

        //Shape shapeIn4D;
        //shapeIn4D.dimensions = {batch_size, 1, 1, input_size};
        TensorDims newDims = {batch_size, 1, 1, input_size};
        //inputDims = newDims;
        input->getTensorDesc().setDims(newDims);
        //dim = convertShapeToDims(shapeIn4D);

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

    mPorts[operation.outputs[0]] = Softmax(input);
    float beta /*scale*/ = PARAM_FP(1);
/*
    if (scale != 1.0f) {
        ALOGE("scale of softmax not suported");
        nnAssert(false);
    }
*/
    VLOG(L1, "Softmax beta = %f ", beta);

    if (beta <= 0.0f) {
        ALOGE("beta must be positive for softmax");
        nnAssert(false);
    }

    return true;
}

bool Executor::operationTANH(const Operation& operation)
{
    VLOG(L1, "OperationType::TANH");
    mPorts[operation.outputs[0]] = Tanh(getPort(operation.inputs[0]));

    return true;
}


void Executor::initializeInput()
{
  VLOG(L1, "initialize Input");
  for (auto i : mModel->inputIndexes)
  {
    int dims_size = mOperands[i].dimensions.size();

/*
    switch(dims_size) {
        case 2:
            mPorts[i]->setLayout(NC);
            break;
        case 4:
            mPorts[i]->setLayout(NCHW);
            break;
        case 1:
            mPorts[i]->setLayout(C);
            break;
        default:
            VLOG(L1, "unsupported dims size %d", dims_size);
            nnAssert(true);
    }
*/
    //mPorts[i]->setPrecision(InferenceEngine::Precision::FP16);



    VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->name.c_str(), dims_size);
    VLOGDIMS(L1, mOperands[i].dimensions, "current operand inpu dims:");
    VLOGDIMS(L1, mPorts[i]->getTensorDesc().getDims(), "Real input dims:");

    auto inputDims = mPorts[i]->getTensorDesc().getDims();

    /*
    for (auto j = 0; j < outputDims.size(); j++)
    VLOG(L1, "output dims[%d] = %d & set output dims[%d] = %d ", j, mOperands[i].dimensions[j], j, outputDims[j]);
    VLOG(L1, "intialization for output data mPorts[%d]->name = %s\n", i, mPorts[i]->name.c_str());
    */

    uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
    auto inputElem = sizeOf(inputDims);
    if (nelem != inputElem) {
      VLOG(L1, "set operand input dims to real input dims\n");
      for (auto j = 0; j < inputDims.size(); j++)
      mOperands[i].dimensions[j] = static_cast<uint32_t>(inputDims[j]);
      mOperands[i].length = sizeOfData(mOperands[i].type, mOperands[i].dimensions);
    }

  }
}


void Executor::finalizeOutput(/*RunTimeOperandInfo* output */)
{
  VLOG(L1, "finalize Output");
  for (auto i : mModel->outputIndexes)
  {
    int dims_size = mOperands[i].dimensions.size();


    switch(dims_size) {
        case 2:
            mPorts[i]->setLayout(NC);
            break;
        case 4:
            mPorts[i]->setLayout(NHWC);
            break;
        case 1:
            mPorts[i]->setLayout(C);
            break;
        default:
            VLOG(L1, "unsupported dims size %d", dims_size);
            nnAssert(true);
    }

    //mPorts[i]->setPrecision(InferenceEngine::Precision::FP16);
    mPorts[i]->setPrecision(InferenceEngine::Precision::FP32);
    mNet.addOutput(mPorts[i]);

    VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->name.c_str(), dims_size);
    VLOGDIMS(L1, mOperands[i].dimensions, "current operand Output dims:");
    VLOGDIMS(L1, mPorts[i]->getTensorDesc().getDims(), "Real Output dims:");

    auto outputDims = mPorts[i]->getTensorDesc().getDims();

    /*
    for (auto j = 0; j < outputDims.size(); j++)
    VLOG(L1, "output dims[%d] = %d & set output dims[%d] = %d ", j, mOperands[i].dimensions[j], j, outputDims[j]);
    VLOG(L1, "intialization for output data mPorts[%d]->name = %s\n", i, mPorts[i]->name.c_str());
    */

    uint32_t nelem = getNumberOfElements(mOperands[i].dimensions);
    auto outputElem = sizeOf(outputDims);
    if (nelem != outputElem) {
      VLOG(L1, "set correct dims as operand output dims different than real output dims\n");
      /*
      for (auto j = 0; j < outputDims.size(); j++)
      mOperands[i].dimensions[j] = static_cast<uint32_t>(outputDims[j]);
      mOperands[i].length = sizeOfData(mOperands[i].type, mOperands[i].dimensions);
      */
    }

  }
}

IRBlob::Ptr VpuExecutor::GetConstWeightsOperandAsTensor(uint32_t index)
{
    dumpOperand(index);
    //const auto op = model.operands.at(index);
    //const auto op = mModel.operands[index];
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, index, len);
    const auto op = mOperands[index];
    VLOG(L1, "VpuExecutor:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
#ifndef MYRIAD_FP32  //Myriad only supprts FP16
        vec<unsigned int> order;
        Layout layout;
        Layout input_layout;
        if (op.dimensions.size() == 4) {
          order = {3,0,1,2};  //IHWO -> OIHW for depth conv
          layout = Layout::OIHW; //weights layout
          input_layout = Layout::NHWC; //same memory layout as OHWI
        }
        else if (op.dimensions.size() == 2) {
          order = {0, 1};
          layout = Layout::NC;
          input_layout = Layout::NC;
        }
        else {
          order = {0}; //(op.dimensions.size() < 2)
          layout = Layout::C;
          input_layout = Layout::C;
        }

        auto inputDims = toDims(op.dimensions);
        TensorDesc td(InferenceEngine::Precision::FP16, inputDims, input_layout);
        // todo: create a readOnly blob that accepts const pointers
        InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
        blob->allocate();
        auto mem = blob->data();
        short *fp16Array = mem.as<short*>();
        // convert from [(float *)buf, len] to fp16Array,
        uint32_t nelem = getNumberOfElements(op.dimensions);
        //short *inputFilter_fp16 = new short[nelem];

        size_t fp16Array_length = nelem*sizeof(short);

        VLOGDIMS(L1, permuteDims(toDims(op.dimensions), order), "weights/bias dims");
        VLOG(L1, "Model buffer oplength = %d bytes nelem= %d fp16Array_length= %d bytes sizeof model buf= %d bytes\n", len , nelem, fp16Array_length, sizeof(buf));

        f32tof16Arrays(fp16Array, (float *)buf, nelem); //OHWI memory layout
        //floattofp16(fp16Array, (float *)buf, nelem);

        if (inputDims.size() != 4) {
              //InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td,(short *)inputFilter_fp16, fp16Array_length);
              return blob;
        } else {
              TensorDesc td(InferenceEngine::Precision::FP16, permuteDims(inputDims, order), layout);
              InferenceEngine::TBlob<short>::Ptr blob_oihw = std::make_shared<InferenceEngine::TBlob<short>>(td);
              blob_oihw->allocate();

              auto dims_ohwi = inputDims; //toDims(op.dimensions);
              size_t out_depth = dims_ohwi[0];
              size_t in_depth = dims_ohwi[3];
              size_t height = dims_ohwi[1];
              size_t width = dims_ohwi[2];
              size_t offset = 0; //blob->size() == o*i*h*w and simlar to nchw memory layout
              //const short* inputFilter = reinterpret_cast<const short *>(buf); //OHWI memory layout

              for (size_t i = 0; i < in_depth; i++){
                for (size_t o = 0; o < out_depth; o++){
                  for (size_t h = 0; h < height; h++){
                    for (size_t w = 0; w < width; w++){
                      size_t offset_ohwi = o*height*width*in_depth + h*width*in_depth + w*in_depth + i; //similar to NHWC memory layout
                      //blob->buffer().as<float*>()[blob->getTensorDesc().offset(offset++)] = inputFilter[offset_ohwi];
                      blob_oihw->buffer().as<short*>()[offset++] = blob->buffer().as<short*>()[offset_ohwi];
                      //size_t offset_oihw = o*in_depth*height*width + i*height*width + h*width + w; //similar to NCHW memory layout
                      //blob->buffer().as<float*>()[offset_oihw] = inputFilter[offset_ohwi];
                      //VLOG(L1, "offset_ohwi= %d offset_oihw= %d", offset_ohwi, offset_oihw);

                    }
                  }
                }
              }

              return blob_oihw;
        }

#else //FP32 support
            vec<unsigned int> order;
            Layout layout;
            if (op.dimensions.size() == 4) {
              order = {3,0,1,2};  //nhwc -> nchw
              layout = Layout::NCHW;
            }
            else if (op.dimensions.size() == 2) {
              order = {0, 1};
              layout = Layout::NC;
            }
            else {
              order = {0}; //(op.dimensions.size() < 2)
              layout = Layout::C;
            }
            TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), order), layout);
            // todo: create a readOnly blob that accepts const pointers
            //return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
#endif
    } else if (op.type == OperandType::TENSOR_INT32) {

            VLOG(L1, "check if const tensors of type IN32 supported");
            TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
            if (buf == nullptr)
                VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
            return blob;
    } else {
            VLOG(L1, "not supporting const tensors of type ", op.type);
            nnAssert(false);
    }
    return nullptr;
}

IRBlob::Ptr VpuExecutor::GetConstOperandAsTensor(uint32_t index)
{
  //const auto op = model.operands.at(index);
  //const auto op = mModel.operands[index];
  const auto op = mOperands[index];
  uint32_t len;
  const uint8_t *buf = GetOperandMemory(mModel, index, len);

  VLOG(L1, "VpuExecutor:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
  if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
#ifndef MYRIAD_FP32  //Myriad only supprts FP16

          vec<unsigned int> order;
          Layout layout;
          Layout input_layout;
          if (op.dimensions.size() == 4) {
            order = {0,3,1,2};  //nhwc -> nchw
            layout = Layout::OIHW; //weights layout
            input_layout = Layout::NHWC; //same memory layout as OHWI
          }
          else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
            input_layout = Layout::NC;
          }
          else {
            order = {0}; //(op.dimensions.size() < 2)
            layout = Layout::C;
            input_layout = Layout::C;
          }

          auto inputDims = toDims(op.dimensions);
          TensorDesc td(InferenceEngine::Precision::FP16, inputDims, input_layout);
          // todo: create a readOnly blob that accepts const pointers
          InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
          blob->allocate();
          auto mem = blob->data();
          short *fp16Array = mem.as<short*>();
          // convert from [(float *)buf, len] to fp16Array,
          uint32_t nelem = getNumberOfElements(op.dimensions);
          //short *inputFilter_fp16 = new short[nelem];

          size_t fp16Array_length = nelem*sizeof(short);

          VLOGDIMS(L1, permuteDims(toDims(op.dimensions), order), "weights/bias dims");
          VLOG(L1, "Model buffer oplength = %d bytes nelem= %d fp16Array_length= %d bytes sizeof model buf= %d bytes\n", len , nelem, fp16Array_length, sizeof(buf));

          f32tof16Arrays(fp16Array, (float *)buf, nelem); //OHWI memory layout
          //floattofp16(fp16Array, (float *)buf, nelem);

          if (inputDims.size() != 4) {
                //InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td,(short *)inputFilter_fp16, fp16Array_length);
                return blob;
          } else {
                TensorDesc td(InferenceEngine::Precision::FP16, permuteDims(inputDims, order), layout);
                InferenceEngine::TBlob<short>::Ptr blob_oihw = std::make_shared<InferenceEngine::TBlob<short>>(td);
                blob_oihw->allocate();

                auto dims_ohwi = inputDims; //toDims(op.dimensions);
                size_t out_depth = dims_ohwi[0];
                size_t in_depth = dims_ohwi[3];
                size_t height = dims_ohwi[1];
                size_t width = dims_ohwi[2];
                size_t offset = 0; //blob->size() == o*i*h*w and simlar to nchw memory layout
                //const short* inputFilter = reinterpret_cast<const short *>(buf); //OHWI memory layout

                for (size_t o = 0; o < out_depth; o++){
                  for (size_t i = 0; i < in_depth; i++){
                    for (size_t h = 0; h < height; h++){
                      for (size_t w = 0; w < width; w++){
                        size_t offset_ohwi = o*height*width*in_depth + h*width*in_depth + w*in_depth + i; //similar to NHWC memory layout
                        //blob->buffer().as<float*>()[blob->getTensorDesc().offset(offset++)] = inputFilter[offset_ohwi];
                        blob_oihw->buffer().as<short*>()[offset++] = blob->buffer().as<short*>()[offset_ohwi];
                        //size_t offset_oihw = o*in_depth*height*width + i*height*width + h*width + w; //similar to NCHW memory layout
                        //blob->buffer().as<float*>()[offset_oihw] = inputFilter[offset_ohwi];
                        //VLOG(L1, "offset_ohwi= %d offset_oihw= %d", offset_ohwi, offset_oihw);

                      }
                    }
                  }
                }

                return blob_oihw;
          }

#else //FP32 support
          vec<unsigned int> order;
          Layout layout;
          if (op.dimensions.size() == 4) {
            order = {0,3,1,2};  //nhwc -> nchw
            layout = Layout::NCHW;
          }
          else if (op.dimensions.size() == 2) {
            order = {0, 1};
            layout = Layout::NC;
          }
          else {
            order = {0}; //(op.dimensions.size() < 2)
            layout = Layout::C;
          }
          TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), order), layout);
          // todo: create a readOnly blob that accepts const pointers
          //return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
#endif
  } else if (op.type == OperandType::TENSOR_INT32) {

          VLOG(L1, "check if const tensors of type IN32 supported");
          TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
          if (buf == nullptr)
              VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
          InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
          return blob;
  } else {
          VLOG(L1, "not supporting const tensors of type ", op.type);
          nnAssert(false);
  }
  return nullptr;
}

Blob::Ptr VpuExecutor::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
  //const auto op = model.operands[index];
  //uint32_t len;
  //const uint8_t *buf = GetOperandMemory(model, index, len);

  if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
#ifndef MYRIAD_FP16  //Myriad supports FP32 only for network input/output
  if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
        VLOG(L1, "Create input blob !!!!");
        vec<unsigned int> order;
        Layout layout;
        if (op.dimensions.size() == 4) {
          order = {0,3,1,2};  //nhwc -> nchw
          layout = Layout::NCHW;
          //layout = Layout::NHWC;
        }
        else if (op.dimensions.size() == 2) {
          order = {0, 1};
          layout = Layout::NC;
        }
        else {
          order = {0}; //(op.dimensions.size() < 2)
          layout = Layout::C;
        }

        auto inputDims = toDims(op.dimensions);
        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
        //TensorDesc td(InferenceEngine::Precision::FP32, inputDims, layout);

        if (buf == nullptr)
            VLOG(L1, "MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");
        if (inputDims.size() != 4) {
              InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
              return blob;
        } else {
              InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
              blob->allocate();

              auto dims_nhwc = inputDims; //toDims(op.dimensions);
              size_t batch = dims_nhwc[0];
              size_t in_depth = dims_nhwc[3]; //channels
              size_t height = dims_nhwc[1];
              size_t width = dims_nhwc[2];
              size_t offset = 0; //blob->size() == o*i*h*w and simlar to nchw memory layout
              const float* input = reinterpret_cast<const float *>(buf); //OHWI memory layout

              //convert NHWC -> NCHW

              for (size_t b = 0; b < batch; b++){
                for (size_t i = 0; i < in_depth; i++){
                  for (size_t h = 0; h < height; h++){
                    for (size_t w = 0; w < width; w++){
                      size_t offset_nhwc = b*height*width*in_depth + h*width*in_depth + w*in_depth + i; //similar to NHWC memory layout
                      blob->buffer().as<float*>()[offset++] = input[offset_nhwc];
                      //blob->buffer().as<float*>()[blob->getTensorDesc().offset(offset++)] = input[offset_nhwc];
                      //size_t offset_nchw = b*in_depth*height*width + i*height*width + h*width + w; //similar to NCHW memory layout
                      //blob->buffer().as<float*>()[offset_oihw] = inputFilter[offset_ohwi];
                      //VLOG(L1, "offset_nhwc= %d offset_nchw= %d", offset_nhwc, offset_nchw);

                    }
                  }
                }
              }

              return blob;
        }
  }
  else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
      VLOG(L1, "Create output blob");
      vec<unsigned int> order;
      Layout layout;
      if (op.dimensions.size() == 4) {
        //order = {0,3,1,2};  //nhwc -> nchw
        layout = Layout::NHWC;
      }
      else if (op.dimensions.size() == 2) {
        //order = {0, 1};
        layout = Layout::NC;
      }
      else {
        //order = {0}; //(op.dimensions.size() < 2)
        layout = Layout::C;
      }

      //TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), Layout::ANY); //nhwc working
      TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout); //nhwc
      //TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), {0,3,1,2}), Layout::ANY);  //nhwc->nchw
          // todo: create a readOnly blob that accepts const pointers
      InferenceEngine::TBlob<float>::Ptr blob = InferenceEngine::make_shared_blob<float>(td, (float *)buf, len);
      return blob;
    }

#else //FP16 support if Myriad does not support FP32 for network input/output

  if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
        // todo: create a readOnly blob that accepts const pointers
    //InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
    vec<unsigned int> order;
    Layout layout;
    if (op.dimensions.size() == 4) {
    order = {0,3,1,2};  //nhwc -> nchw
    layout = Layout::NCHW;
    }
    else if (op.dimensions.size() == 2) {
    order = {0, 1};
    layout = Layout::NC;
    }
    else {
    order = {0}; //(op.dimensions.size() < 2)
    layout = Layout::C;
    }
    TensorDesc td(InferenceEngine::Precision::FP16, permuteDims(toDims(op.dimensions), order), layout);
    InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);

    //InferenceEngine::TBlob<short>::Ptr blob = InferenceEngine::make_shared_blob<short, InferenceEngine::SizeVector>(InferenceEngine::Precision::FP16, permuteDims(toDims(op.dimensions), order));

    blob->allocate();
    auto mem = blob->data();
    short *fp16Array = mem.as<short*>();
    // convert from [(float *)buf, len] to fp16Array, blob->size()
  uint32_t nelem = getNumberOfElements(op.dimensions);
  VLOG(L1, "Model buffer oplength = %d bytes nelem= %d fp16Array= %d bytes sizeof model buf= %d bytes\n", len , nelem, sizeof(fp16Array), sizeof(buf));
  if (blob->size() != nelem) {
  VLOG(L1, "Model buffer len = %d bytes nelem= %d fp16Array= %d bytes\n",len , nelem, sizeof(fp16Array));
  nnAssert(false);
  }

  if (buf == nullptr) {
  VLOG(L1, "Request model input buffer is null pointer");
  }

  f32tof16Arrays(fp16Array, (float *)buf, nelem);

  return blob;
  }
  else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {

    vec<unsigned int> order;
    Layout layout;
    if (op.dimensions.size() == 4) {
      //order = {0,3,1,2};  //nhwc -> nchw
      layout = Layout::NHWC;
    }
    else if (op.dimensions.size() == 2) {
      //order = {0, 1};
      layout = Layout::NC;
    }
    else {
      //order = {0}; //(op.dimensions.size() < 2)
      layout = Layout::C;
    }

    TensorDesc td(InferenceEngine::Precision::FP16, toDims(op.dimensions), layout);
        // todo: create a readOnly blob that accepts const pointers
    InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
    blob->allocate();
    auto mem = blob->data();
    short *fp16Array = mem.as<short*>();
    // convert from [(float *)buf, len] to fp16Array, blob->size()
    uint32_t length = len/sizeof(short);
    VLOG(L1, "Model buffer len = %d bytes length= %d bytes fp16Array= %d bytes\n",len , length, sizeof(fp16Array));
    if (length >= sizeof(fp16Array)) {
    VLOG(L1, "Model buffer len = %d bytes length= %d bytes fp16Array= %d bytes\n",len , length, sizeof(fp16Array));
    nnAssert(false);
    }
    //need memcpy after infer output ??
    f16tof32Arrays((float *)buf, fp16Array,length);

    return blob;
  }

      //TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), Layout::ANY);
      // todo: create a readOnly blob that accepts const pointers
      //return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
#endif
  }
  else if (op.type == OperandType::TENSOR_INT32) {
      VLOG(L1, "check if const tensors of type IN32 supported");
      //nnAssert(true);
      //TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
      //return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t *)buf, len);
  } else {

      VLOG(L1, "not supporting const tensors of type ", op.type);
      nnAssert(false);
  }
  return nullptr;

}

IRBlob::Ptr CpuExecutor::GetConstWeightsOperandAsTensor(uint32_t index)
{
    dumpOperand(index);
    //const auto op = mModel.operands[index];
    const auto op = mOperands[index];
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, index, len);
    VLOG(L1, "CpuExecutor:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
      vec<unsigned int> order;
      Layout layout;
      if (op.dimensions.size() == 4) {
        //order = {0,3,1,2};  //nhwc -> nchw
        order = {3,0,1,2};  //IHWO -> OIHW for depth conv
        //layout = Layout::NCHW;
        //layout = Layout::NHWC;
        layout = Layout::OIHW; //weights layout
      }
      else if (op.dimensions.size() == 2) {
        order = {0, 1};
        layout = Layout::NC;
      }
      else {
        order = {0}; //(op.dimensions.size() < 2)
        layout = Layout::C;
      }
      auto inputDims = toDims(op.dimensions);
      TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
      //TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout);
      if (buf == nullptr)
          VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
      if (inputDims.size() != 4) {
            InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
            return blob;
      } else {
            InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();

            auto dims_ohwi = inputDims; //toDims(op.dimensions);
            size_t out_depth = dims_ohwi[0];
            size_t in_depth = dims_ohwi[3];
            size_t height = dims_ohwi[1];
            size_t width = dims_ohwi[2];
            size_t offset = 0; //blob->size() == o*i*h*w and simlar to nchw memory layout
            const float* inputFilter = reinterpret_cast<const float *>(buf); //OHWI memory layout

            //convert OHWI -> OIHW

            //for depth conv need reorder as IOHW since for tflite O is always 1 and IE expects reorder to
            //[in_channels, depth_multiplier, filter_height, filter_width]
            for (size_t i = 0; i < in_depth; i++){
              for (size_t o = 0; o < out_depth; o++){
                for (size_t h = 0; h < height; h++){
                  for (size_t w = 0; w < width; w++){
                    size_t offset_ohwi = o*height*width*in_depth + h*width*in_depth + w*in_depth + i; //similar to NHWC memory layout
                    blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                    //blob->buffer().as<float*>()[blob->getTensorDesc().offset(offset++)] = inputFilter[offset_ohwi];
                    //size_t offset_oihw = o*in_depth*height*width + i*height*width + h*width + w; //similar to NCHW memory layout
                    //blob->buffer().as<float*>()[offset_oihw] = inputFilter[offset_ohwi];
                    //VLOG(L1, "offset_ohwi= %d offset_oihw= %d", offset_ohwi, offset_oihw);

                  }
                }
              }
            }

            return blob;
      }
    } else if (op.type == OperandType::TENSOR_INT32) {

        VLOG(L1, "check if const tensors of type IN32 supported");
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
        if (buf == nullptr)
            VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
        InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
        return blob;
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}


IRBlob::Ptr CpuExecutor::GetConstOperandAsTensor(uint32_t index)
{
  dumpOperand(index);
  //const auto op = mModel.operands[index];
  const auto op = mOperands[index];
  uint32_t len;
  const uint8_t *buf = GetOperandMemory(mModel, index, len);
  VLOG(L1, "CpuExecutor:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
  if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
      vec<unsigned int> order;
      Layout layout;
      if (op.dimensions.size() == 4) {
        order = {0,3,1,2};  //nhwc -> nchw
        //layout = Layout::NCHW;
        //layout = Layout::NHWC;
        layout = Layout::OIHW; //weights layout
      }
      else if (op.dimensions.size() == 2) {
        order = {0, 1};
        layout = Layout::NC;
      }
      else {
        order = {0}; //(op.dimensions.size() < 2)
        layout = Layout::C;
      }
      auto inputDims = toDims(op.dimensions);
      TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
      //TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout);
      if (buf == nullptr)
          VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
      if (inputDims.size() != 4) {
            InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
            return blob;
      } else {
            InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
            blob->allocate();

            auto dims_ohwi = inputDims; //toDims(op.dimensions);
            size_t out_depth = dims_ohwi[0];
            size_t in_depth = dims_ohwi[3];
            size_t height = dims_ohwi[1];
            size_t width = dims_ohwi[2];
            size_t offset = 0; //blob->size() == o*i*h*w and simlar to nchw memory layout
            const float* inputFilter = reinterpret_cast<const float *>(buf); //OHWI memory layout

            for (size_t o = 0; o < out_depth; o++){
              for (size_t i = 0; i < in_depth; i++){
                for (size_t h = 0; h < height; h++){
                  for (size_t w = 0; w < width; w++){
                    size_t offset_ohwi = o*height*width*in_depth + h*width*in_depth + w*in_depth + i; //similar to NHWC memory layout
                    //blob->buffer().as<float*>()[blob->getTensorDesc().offset(offset++)] = inputFilter[offset_ohwi];
                    blob->buffer().as<float*>()[offset++] = inputFilter[offset_ohwi];
                    //size_t offset_oihw = o*in_depth*height*width + i*height*width + h*width + w; //similar to NCHW memory layout
                    //blob->buffer().as<float*>()[offset_oihw] = inputFilter[offset_ohwi];
                    //VLOG(L1, "offset_ohwi= %d offset_oihw= %d", offset_ohwi, offset_oihw);

                  }
                }
              }
            }

            return blob;
      }

  } else if (op.type == OperandType::TENSOR_INT32) {

      VLOG(L1, "check if const tensors of type IN32 supported");
      TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
      if (buf == nullptr)
          VLOG(L1, "TENSOR_INT32 buf is NULL !!!!!!!!!!!!!!!");
      InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
      return blob;
  } else {
      VLOG(L1, "not supporting const tensors of type ", op.type);
      nnAssert(false);
  }
  return nullptr;
}

Blob::Ptr CpuExecutor::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
  if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
      if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
          VLOG(L1, "Create input blob !!!!");
          vec<unsigned int> order;
          Layout layout;
          if (op.dimensions.size() == 4) {
          order = {0,3,1,2};  //nhwc -> nchw
          layout = Layout::NCHW;
          //layout = Layout::NHWC;
          }
          else if (op.dimensions.size() == 2) {
          order = {0, 1};
          layout = Layout::NC;
          }
          else {
          order = {0}; //(op.dimensions.size() < 2)
          layout = Layout::C;
          }

          auto inputDims = toDims(op.dimensions);
          TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(inputDims, order), layout);
          //TensorDesc td(InferenceEngine::Precision::FP32, inputDims, layout);

          if (buf == nullptr)
              VLOG(L1, "MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");
          if (inputDims.size() != 4) {
                InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
                return blob;
          } else {
                InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td);
                blob->allocate();

                auto dims_nhwc = inputDims; //toDims(op.dimensions);
                size_t batch = dims_nhwc[0];
                size_t in_depth = dims_nhwc[3]; //channels
                size_t height = dims_nhwc[1];
                size_t width = dims_nhwc[2];
                size_t offset = 0; //blob->size() == o*i*h*w and simlar to nchw memory layout
                const float* input = reinterpret_cast<const float *>(buf); //OHWI memory layout

                //convert NHWC -> NCHW

                for (size_t b = 0; b < batch; b++){
                  for (size_t i = 0; i < in_depth; i++){
                    for (size_t h = 0; h < height; h++){
                      for (size_t w = 0; w < width; w++){
                        size_t offset_nhwc = b*height*width*in_depth + h*width*in_depth + w*in_depth + i; //similar to NHWC memory layout
                        blob->buffer().as<float*>()[offset++] = input[offset_nhwc];
                        //blob->buffer().as<float*>()[blob->getTensorDesc().offset(offset++)] = input[offset_nhwc];
                        //size_t offset_nchw = b*in_depth*height*width + i*height*width + h*width + w; //similar to NCHW memory layout
                        //blob->buffer().as<float*>()[offset_oihw] = inputFilter[offset_ohwi];
                        //VLOG(L1, "offset_nhwc= %d offset_nchw= %d", offset_nhwc, offset_nchw);

                      }
                    }
                  }
                }

                return blob;
          }
      } else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
          VLOG(L1, "Create output blob !!!!");
          vec<unsigned int> order;
          Layout layout;
          if (op.dimensions.size() == 4) {
            //order = {0,3,1,2};  //nhwc -> nchw
            layout = Layout::NHWC;
          }
          else if (op.dimensions.size() == 2) {
            //order = {0, 1};
            layout = Layout::NC;
          }
          else {
            //order = {0}; //(op.dimensions.size() < 2)
            layout = Layout::C;
          }

          TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), layout); //nhwc
          if (buf == nullptr)
              VLOG(L1, "MODEL_OUTPUT buf is NULL !!!!!!!!!!!!!!!");
          InferenceEngine::TBlob<float>::Ptr blob = InferenceEngine::make_shared_blob<float>(td, (float *)buf, len);
          return blob;
      }
  } else if (op.type == OperandType::TENSOR_INT32) {
      VLOG(L1, "check if const tensors of type IN32 supported");
      //nnAssert(true);
      TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
      return std::make_shared<InferenceEngine::TBlob<int32_t>>(td, (int32_t *)buf, len);
  } else {
      VLOG(L1, "not supporting const tensors of type ", op.type);
      nnAssert(false);
  }
  return nullptr;
}

bool PreparedModel::validateRequest(const Request& request, const Model& model)
{
    auto validRequestArguments = [&](const hidl_vec<RequestArgument>& arguments,
                                     const hidl_vec<uint32_t>& operandIndexes,
                                     const hidl_vec<Operand>& operands, size_t poolCount,
                                     const char* type) -> bool {
        const size_t argumentCount = arguments.size();
        if (argumentCount != operandIndexes.size()) {
            ALOGE("Request specifies %zu, but %s has the model as %zu", argumentCount, type,
                  operandIndexes.size());
            return false;
        }
        for (size_t argumentIndex = 0; argumentIndex < argumentCount; argumentIndex++) {
            const RequestArgument& argument = arguments[argumentIndex];
            const uint32_t operandIndex = operandIndexes[argumentIndex];
            const Operand& operand = operands[operandIndex];
            if (argument.hasNoValue) {
                if (argument.location.poolIndex != 0 ||
                    argument.location.offset != 0 ||
                    argument.location.length != 0 ||
                    argument.dimensions.size() != 0) {
                    ALOGE("Request %s: %zu has no value yet has details.", type, argumentIndex);
                    return false;
                }
            }
            if (argument.location.poolIndex >= poolCount) {
                ALOGE("Request %s: %zu has an invalid poolIndex %d", type, argumentIndex,
                      argument.location.poolIndex);
                return false;
            }
            // TODO: Validate that we are within the pool.
            uint32_t rank = argument.dimensions.size();
            if (rank > 0) {
                if (rank != operand.dimensions.size()) {
                    ALOGE("Request %s: %zu  has number of dimensions (%d) different than the model's (%zu)",
                          type, argumentIndex, rank, operand.dimensions.size());
                    return false;
                }
                for (size_t i = 0; i < rank; i++) {
                    if (argument.dimensions[i] != operand.dimensions[i] &&
                        operand.dimensions[i] != 0) {
                        ALOGE("Request %s: %zu has dimension %zu of %d different than the model's %d",
                              type, argumentIndex, i, operand.dimensions[i], operand.dimensions[i]);
                        return false;
                    }
                    if (argument.dimensions[i] == 0) {
                        ALOGE("Request %s: %zu has dimension %zu of zero", type, argumentIndex, i);
                        return false;
                    }
                }
            }
        }
        return true;
    };

    const size_t pool_counts = request.pools.size();
    return (validRequestArguments(request.inputs, model.inputIndexes, model.operands, pool_counts,
                                  "input") &&
            validRequestArguments(request.outputs, model.outputIndexes, model.operands, pool_counts,
                                  "output"));
}

bool PreparedModel::validModel(const Model& model)
{
    auto validOperandIndexes = [&](const hidl_vec<uint32_t> indexes, size_t operandCount) -> bool {
        for (uint32_t i : indexes) {
            if (i >= operandCount) {
                ALOGE("Index out of range %d/%zu",i , operandCount);
                return false;
            }
        }
        return true;
    };

    const size_t operand_counts = model.operands.size();

    if (!validOperandIndexes(model.inputIndexes, operand_counts) ||
        !validOperandIndexes(model.outputIndexes, operand_counts)) {
        ALOGE("model inputs/outputs index invalid");
        return false;
    }

    for (size_t i = 0; i < operand_counts; i++) {
        const Operand& operand = model.operands[i];
        if (operand.type != OperandType::TENSOR_FLOAT32 &&
            operand.type != OperandType::FLOAT32 &&
            operand.type != OperandType::INT32 &&
            operand.type != OperandType::UINT32 &&
            operand.type != OperandType::TENSOR_INT32 &&
            operand.type != OperandType::TENSOR_QUANT8_ASYMM) {
            ALOGE("wrong operand type %d", operand.type);
            return false;
         }

        switch (operand.lifetime) {
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
            case OperandLifeTime::TEMPORARY_VARIABLE:
                if (operand.location.offset != 0 || operand.location.length != 0) {
                    ALOGE("Unexpected offset %d, or length %d",
                          operand.location.offset, operand.location.length);
                    return false;
                }
                break;
            case OperandLifeTime::CONSTANT_COPY:
                if (operand.location.offset + operand.location.length > model.operandValues.size()) {
                    ALOGE("OperandValue location out of range.  Starts at %d length %d. max %zu",
                          operand.location.offset, operand.location.length, model.operandValues.size());
                    return false;
                }
                break;
            case OperandLifeTime::CONSTANT_REFERENCE:
            {
                auto pool_counts = model.pools.size();
                if (operand.location.poolIndex >= pool_counts) {
                    ALOGE("Invalid poolIndex %d/%lu", operand.location.poolIndex, pool_counts);
                    return false;
                }
                break;
            }
            default:
                return false;
        }
    }

    for (const auto& operation : model.operations) {
        //TANH is last one
        if (static_cast<uint32_t>(operation.type) >
            static_cast<uint32_t>(OperationType::TANH)) {
            return false;
        }
    }

    return true;
}


bool PreparedModel::isOperationSupported(const Operation& operation, const Model& model)
{
    VLOG(L1, "Check operation %d", operation.type);

    #define VLOG_CHECKFAIL(fail)  VLOG(L1, "Check failed: %s", fail)

#ifdef DISABLE_ALL_QUANT
    for (auto i : operation.inputs) {
        const auto input = model.operands[i];
        if (input.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("input quant");
            return false;
        }
    }
    for (auto i : operation.outputs) {
        const auto output = model.operands[i];
        if (output.type == OperandType::TENSOR_QUANT8_ASYMM) {
            VLOG_CHECKFAIL("output quant");
            return false;
        }
    }
#else
    for (auto i : operation.inputs) {
        const auto input = model.operands[i];
        if (input.type == OperandType::TENSOR_QUANT8_ASYMM &&
            input.zeroPoint != 0) {
            VLOG_CHECKFAIL("input quant");
            return false;
        }
    }
    for (auto i : operation.outputs) {
        const auto output = model.operands[i];
        if (output.type == OperandType::TENSOR_QUANT8_ASYMM &&
            output.zeroPoint != 0) {
            VLOG_CHECKFAIL("output quant");
            return false;
        }
    }
#endif

    const auto input0 = model.operands[operation.inputs[0]];
    auto activationPass = [&model](const Operand& input) -> bool {
        const FusedActivationFunc activation = getOperandConstVal<FusedActivationFunc>(&model, input);
        /*
        if (activation == FusedActivationFunc::RELU1) {
            VLOG_CHECKFAIL("relu1 used");
            return false;
        }
        */
        return true;
    };

    const auto& inputn = model.operands[operation.inputs[operation.inputs.size() - 1]];

    switch(operation.type) {

        case OperationType::CONV_2D:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            //filter in == channel
            if (input0.dimensions[3] != input1.dimensions[3]) {
                VLOG_CHECKFAIL("filter in not equals channel");
                return false;
            }
            break;
            //continue to check actication.
        }

        case OperationType::RELU: break;
        case OperationType::RELU1: break;
        case OperationType::RELU6: break;

        case OperationType::DEPTHWISE_CONV_2D:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            //channels_out must be channels * depth_mul
            if ((input1.dimensions[3] % input0.dimensions[3]) != 0) {
                VLOG_CHECKFAIL("dims not in group");
                return false;
            }
            if (activationPass(inputn) == false) {
                return false;
            }
            break;
        }


        case OperationType::SOFTMAX:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            float beta = getOperandConstVal<float>(&model, input1);
            //beta need = 1.0f
            //if (beta != 1.0f) {
            if (beta <= 0.0f) {
                VLOG_CHECKFAIL("beta must be positive for softmax");
                return false;
            }

            break;
        }

        case OperationType::AVERAGE_POOL_2D: break;

        case OperationType::MAX_POOL_2D:
        case OperationType::FULLY_CONNECTED:
        {
            if (activationPass(inputn) == false) {
                return false;
            }
            break;
        }

        case OperationType::RESHAPE:
             break;
/*
        case OperationType::LOGISTIC:
        case OperationType::TANH:
        case OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case OperationType::CONCATENATION:
        case OperationType::L2_NORMALIZATION:
        case OperationType::ADD: {
           const auto& input1 = model.operands[operation.inputs[1]];
           if (input0.dimensions != input1.dimensions) {
               VLOG_CHECKFAIL("dims not match");
               return false;
           }

           if (activationPass(inputn) == false) {
               return false;
           }
           break;
        } */
        default:
           VLOG(L1, "unsupport opration %d", operation.type);
           return false;
    }
    VLOG(L1, "Operation %d supported by driver", operation.type);

    return true;
}

void PreparedModel::asyncExecute(const Request& request,
                                       const sp<IExecutionCallback>& callback)
{
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!executor::setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        callback->notify(ErrorStatus::GENERAL_FAILURE);
        return;
    }

    if (mTargetDevice == TargetDevice::eCPU){
      CpuExecutor executor;
      int n = executor.run(mModel, request, mPoolInfos, requestPoolInfos);
    }else if (mTargetDevice == TargetDevice::eMYRIAD) {
      VpuExecutor executor;
      int n = executor.run(mModel, request, mPoolInfos, requestPoolInfos);
    }


    Return<void> returned = callback->notify(ErrorStatus::NONE);
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }


}


Return<ErrorStatus> PreparedModel::execute(const Request& request,
                                                 const sp<IExecutionCallback>& callback)
{

    VLOG(L1, "Begin to execute");
/*
    if (mPorts.size() == 0) {
        ALOGE("No primitive to execute");
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }
*/
    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (validateRequest(request, mModel) == false) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }


    // This thread is intentionally detached because the vpu driver service
    // is expected to live forever.
    std::thread([this, request, callback]{ asyncExecute(request, callback); }).detach();

    VLOG(L1, "Start execute thread done");

    return ErrorStatus::NONE;
}


bool PreparedModel::initialize() {
    return executor::setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
}


}  // namespace executor
}  // namespace driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
