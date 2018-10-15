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

#define LOG_TAG "PreparedModel"


#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <thread>
#include <fstream>
#include "PreparedModel.h"

#define DISABLE_ALL_QUANT
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
        const auto op = mModel.operands[index];                         \
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


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace driver {


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
T PreparedModel::ParseOperationInput(const Model& model, const Operation& operation, uint32_t index)
{
    uint32_t inputIndex = operation.inputs[index];
    const auto operand = mModel.operands[inputIndex];
    const auto value = GetConstOperand<T>(model, inputIndex);
    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
    VLOG(L1, "Operation input index: %d, operand index: %d", index, inputIndex);
    VLOG(L1, "Operation: %s", toString(operation).c_str());
    //VLOG(L1, "Operand: value: %d, %s", alue, toString(operand).c_str());
    printHelper<T>::print(value, toString(operand).c_str());
    VLOG(L1, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

    return value;
}

OutputPort PreparedModel::handleFusion(const OutputPort &out, int32_t fusedOp)
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
T PreparedModel::GetConstFromBuffer(const uint8_t *buf, uint32_t len) {
    VLOG(L1, "buf: %p, len: %d", buf, len);
    if (len != sizeof(T)) {
        VLOG(L1, "typeid(T).name() should be %d bytes", sizeof(T));
        nnAssert(false);
    }
    return *(T *)(buf);
}


template<typename T>
std::vector<T> PreparedModel::GetConstVecFromBuffer(const uint8_t *buf, uint32_t len) {
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

const uint8_t* PreparedModel::GetOperandMemory(const Model &model, uint32_t index, uint32_t &len_out)
{
    const auto op = model.operands[index];
    len_out = op.location.length;
    if (op.lifetime == OperandLifeTime::CONSTANT_COPY)
    {
        if (op.location.poolIndex != 0){
            ALOGE("CONSTANT_COPY expects poolIndex to be 0");
            nnAssert(false);
        //return &model.operandValues[op.location.offset];
        }
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_COPY");
        return (const_cast<uint8_t*>(&model.operandValues[op.location.offset]));
      //to.numberOfUsesLeft = 0;
    } else if (op.lifetime == OperandLifeTime::CONSTANT_REFERENCE)
    {
        //auto pool = model.pools[op.location.poolIndex];
        //return (pool + op.location.offset);
        VLOG(L1, "operand lifetime OperandLifeTime::CONSTANT_REFERENCE");
        auto poolIndex = op.location.poolIndex;
        //nnAssert(poolIndex < mPoolInfos.size()); //aks fix me
        auto& r = mPoolInfos[poolIndex];
        return (const_cast<uint8_t*>(r.buffer + op.location.offset));
        //to.numberOfUsesLeft = 0;
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


    ALOGE("operand is expected to be const, but lifetime is %d", op.lifetime);
    nnAssert(false);
    return nullptr;
}

template <typename T>
T PreparedModel::GetConstOperand(const Model &model, uint32_t index)
{
    dumpOperand(index);
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(model, index, len);
    return GetConstFromBuffer<T>(buf, len);
}

template <typename T>
std::vector<T> PreparedModel::GetConstVecOperand(const Model &model, uint32_t index)
{
    dumpOperand(index);
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(model, index, len);
    return GetConstVecFromBuffer<T>(buf, len);
}

//#define MYRIAD_FP32

IRBlob::Ptr PreparedModel::GetConstOperandAsTensor(uint32_t index)
{
    return nullptr;
}

Blob::Ptr PreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
    return nullptr;
}

OutputPort PreparedModel::getPort(int index)
{
  VLOG(L1, "getPort\n");
	if (isConst(index))
	{
		VLOG(L1, "index is a const!");
		nnAssert(false);
	}
	const auto op = mModel.operands[index];
	if (op.lifetime == OperandLifeTime::MODEL_INPUT)
	{
          VLOG(L1, "Model input operand\n");
          std::ostringstream operandName; operandName << "operand."<<index;

          vec<unsigned int> order;
          if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
          else if (op.dimensions.size() == 2) order = {0, 1};
          else order = {0}; //(op.dimensions.size() < 2)

          auto operandInfo = mNet.createInput(operandName.str(), permuteDims(toDims(op.dimensions), order)); // NHWC -> NCHW
          mPorts[index] = operandInfo->getInputData();
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

void PreparedModel::SetOperandMemory(const Model &model, uint32_t index, uint32_t &len_out, const uint8_t *buf)
{

}

/*
void PreparedModel::SetOperandFromTensor(uint8_t* buf, uint32_t &length, IRBlob::Ptr infOutput)
{

    auto& td = infOutput->getTensorDesc();
    auto dims = infOutput->getTensorDesc().getDims();
    if (td.getPrecision() == InferenceEngine::Precision::FP16)
        td.setPrecision(InferenceEngine::Precision::FP16);
    if (td.getLayout() == Layout::NCHW)
        td.setLayout(NHWC);
    auto mem = infOutput->readOnly();

    const float *pf = mem.as<const float*>();

    length = infOutput->size();
    buf = (uint8_t*) pf;


    return;
}
*/
// TODO doublecheck
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


bool PreparedModel::initializeRunTimeOperandInfo() {
    //initialize runtime operand info from model.
    const size_t count = mModel.operands.size();
    mOperands.resize(count);
    mPorts.resize(count);
    //TensorDims dims;

    // Start by setting the runtime info to what's in the model.
    for (size_t i = 0; i < count; i++) {
        const Operand& from = mModel.operands[i];
        RunTimeOperandInfo& to = mOperands[i];
//        OutputPort& port = mPorts[i];  //std::shared_ptr<Data>
        to.dimensions.resize(from.dimensions.size());
        //dims.resize(from.dimensions.size());
        //port->setDims(from.dimensions, NHWC);  //std::vector<size_t>  //std::vector<uint32_t>
        for (size_t j = 0; j < from.dimensions.size(); j++) {
            to.dimensions[j] = from.dimensions[j];
            //dims[j] = (size_t)from.dimensions[j];
        }
//        to.opIdx = i; //Fix Me: temp setting dataId of operand
//        to.name = "tensor" + std::to_string(i);  //Fix Me: temp setting name of operand

        //auto input = mPorts[i] = mNet.createInput("input", dims);
        //set default input layout
        //input->setLayout(NHWC);

        to.scale = from.scale;
        nnAssert(from.zeroPoint == 0);
        switch(from.type) {
            case OperandType::TENSOR_FLOAT32:
            case OperandType::FLOAT32:
                //nnAssert(to.scale == 0);
                to.type = OperandType::TENSOR_FLOAT32;
                VLOG(L1, "OperandType = %d\n", from.type);
                //port->setPrecision(InferenceEngine::Precision::FP32);
                //to.type = VpuDataType::FP32;
                //to.type = InferenceEngine::VPU::VpuDataType::FP32;
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
        }

        to.length = from.location.length;
        to.lifetime = from.lifetime;
        switch (from.lifetime) {
            case OperandLifeTime::TEMPORARY_VARIABLE:
                to.buffer = nullptr;
                to.length = sizeOfData(to.type, to.dimensions);
                to.numberOfUsesLeft = from.numberOfConsumers;
                break;
            case OperandLifeTime::CONSTANT_COPY:
                to.buffer = const_cast<uint8_t*>(&mModel.operandValues[from.location.offset]);
                to.numberOfUsesLeft = 0;
                break;
            case OperandLifeTime::CONSTANT_REFERENCE:
            {
                auto poolIndex = from.location.poolIndex;
                nnAssert(poolIndex < mPoolInfos.size());
                auto& r = mPoolInfos[poolIndex];
                to.buffer = r.buffer + from.location.offset;
                to.numberOfUsesLeft = 0;
                break;
            }
            case OperandLifeTime::MODEL_INPUT:
            case OperandLifeTime::MODEL_OUTPUT:
            case OperandLifeTime::NO_VALUE:
                to.buffer = nullptr;
                to.length = sizeOfData(to.type, to.dimensions);
                to.numberOfUsesLeft = 0;
                break;
            default:
                return false;
                break;
        }
    }
    return true;
}

/*
bool PreparedModel::initialize() {
    return setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
}
*/
bool PreparedModel::initialize()
{
    VLOG(L1, "initialize");
    bool success = false;

    //Check operation supoorted or not, user may not call getOpertionSupported()
    for (const auto& operation : mModel.operations) {
        success = isOperationSupported(operation, mModel);
        dumpOperationSupport(operation,success);
        if (!success) {
            VLOG(L1, "get unsupported operation in initialize()");
            return false;
        }
    }

    success = setRunTimePoolInfosFromHidlMemories(&mPoolInfos, mModel.pools);
    if (!success) {
        VLOG(L1, "setRunTimePoolInfosFromHidlMemories failed.");
        return false;
    }

    success = initializeRunTimeOperandInfo();


    if (!success) {
        VLOG(L1, "initializeRunTimeOperandInfo failed.");
        return false;
    }


    for (const auto& operation : mModel.operations) {
        VLOG(L1, "get operation %d ready to add", operation.type);
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
    }

    finalizeOutput();

    //initialize IE operation input/output ports
//    convertModel(mNet);

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
    return true;
}

void PreparedModel::deinitialize()
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

void PreparedModel::asyncExecute(const Request& request,
                                       const sp<IExecutionCallback>& callback)
{

    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        callback->notify(ErrorStatus::GENERAL_FAILURE);
        return;
    }

    //std::vector<IRBlob::Ptr> input;
    //std::vector<TBlob<float>::Ptr> output;
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
                //auto in = GetOperandAsTensor(operand, operand.buffer, operand.length);
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



    // compile it

    //InfEng inference_engine(InferenceEngine::TargetDevice::eMYRIAD);

    //auto executeNet1 = inference_engine.Compile(mNet);

    VLOG(L1, "pass request inputs/outputs buffer to network/model respectively");

    inOutData(mModel.inputIndexes, request.inputs, true, enginePtr, mPorts);
    inOutData(mModel.outputIndexes, request.outputs, false, enginePtr, mPorts);

    VLOG(L1, "Run");

    //auto output = execute.Infer(input).wait();
    enginePtr->Infer();


//    VLOG(L1, "copy model output to request output");

    VLOG(L1, "update shared memories");
    for (auto runtimeInfo : requestPoolInfos) {
        runtimeInfo.update();
    }

#ifdef NN_DEBUG
    {
        VLOG(L1, "Model output0 are:");
        const RunTimeOperandInfo& output = mOperands[mModel.outputIndexes[0]];
        InferenceEngine::TBlob<float>::Ptr outBlob = enginePtr->getBlob(mPorts[mModel.outputIndexes[0]]->name);

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
        const RunTimeOperandInfo& input = mOperands[mModel.inputIndexes[0]];
        InferenceEngine::TBlob<float>::Ptr inBlob = enginePtr->getBlob(mPorts[mModel.inputIndexes[0]]->name);
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
        for(const auto& op : mModel.operations) {
            const auto& o = mOperands[op.outputs[0]];
            InferenceEngine::TBlob<float>::Ptr opBlob = enginePtr->getBlob(mPorts[op.outputs[0]]->name);
            VLOG(L1, "Operation %d has output 0(lifetime %d) are:", op.type, o.lifetime);

            nelem = (opBlob->size() > 10 ? 10 : opBlob->size());
            for (int i = 0; i <  nelem; i++) {
            VLOG(L1, "operation output Blob elements %d = %f", i, opBlob->readOnly()[i]);
            }
            //printOperandbuf(L4, o.buffer, o.dimensions, 20);
        }
    }
#endif

    Return<void> returned = callback->notify(ErrorStatus::NONE);
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }


}

Return<ErrorStatus> PreparedModel::execute(const Request& request,
                                                 const sp<IExecutionCallback>& callback)
{

    VLOG(L1, "Begin to execute");

    if (mPorts.size() == 0) {
        ALOGE("No primitive to execute");
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

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

/*
void PreparedModel::asyncExecute(const Request& request,
                                       const sp<IExecutionCallback>& callback) {
    std::vector<RunTimePoolInfo> requestPoolInfos;
    if (!setRunTimePoolInfosFromHidlMemories(&requestPoolInfos, request.pools)) {
        callback->notify(ErrorStatus::GENERAL_FAILURE);
        return;
    }

    VpuExecutor executor;
    int n = executor.run(mModel, request, mPoolInfos, requestPoolInfos);
//    VLOG(DRIVER) << "executor.run returned " << n;
    VLOG(L1, "executor.run returned %d", n);
    ErrorStatus executionStatus =
            n == ANEURALNETWORKS_NO_ERROR ? ErrorStatus::NONE : ErrorStatus::GENERAL_FAILURE;
    Return<void> returned = callback->notify(executionStatus);
    if (!returned.isOk()) {
        LOG(ERROR) << " hidl callback failed to return properly: " << returned.description();
    }
}

Return<ErrorStatus> PreparedModel::execute(const Request& request,
                                                 const sp<IExecutionCallback>& callback) {
//    VLOG(DRIVER) << "execute(" << toString(request) << ")";
    if (callback.get() == nullptr) {
        LOG(ERROR) << "invalid callback passed to execute";
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, mModel)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the vpu driver service
    // is expected to live forever.
    std::thread([this, request, callback]{ asyncExecute(request, callback); }).detach();

    return ErrorStatus::NONE;
}
*/

template <typename T>
T getOperandConstVal(const Model& model, const Operand& operand)
{
    const T* data = reinterpret_cast<const T *>(&model.operandValues[operand.location.offset]);
    return data[0];
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
        const FusedActivationFunc activation = getOperandConstVal<FusedActivationFunc>(model, input);
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
            float beta = getOperandConstVal<float>(model, input1);
            //beta need = 1.0f
            if (beta != 1.0f) {
                VLOG_CHECKFAIL("beta not 1.0f");
                return false;
            }
            break;
        }
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
        case OperationType::AVERAGE_POOL_2D:
        case OperationType::MAX_POOL_2D:
        case OperationType::FULLY_CONNECTED:
        {
            if (activationPass(inputn) == false) {
                return false;
            }
            break;
        }
        case OperationType::ADD:
        {
            const auto& input1 = model.operands[operation.inputs[1]];
            if (input0.dimensions != input1.dimensions) {
                VLOG_CHECKFAIL("dims not match");
                return false;
            }

            if (activationPass(inputn) == false) {
                return false;
            }
            break;
        }
        case OperationType::RELU:
        case OperationType::RELU1:
        case OperationType::RELU6:
        case OperationType::LOGISTIC:
        case OperationType::TANH:
        case OperationType::LOCAL_RESPONSE_NORMALIZATION:
        case OperationType::CONCATENATION:
        case OperationType::L2_NORMALIZATION:
        case OperationType::RESHAPE:
             break;
        default:
           VLOG(L1, "unsupport opration %d", operation.type);
           return false;
    }
    VLOG(L1, "Operation %d supported by driver", operation.type);

    return true;
}

bool PreparedModel::isConst(int index)
{
    VLOG(L1, "---------------------------------------------");
    VLOG(L1, "Operand index: %d", index);
    const auto op = mModel.operands[index];
    VLOG(L1, " %s", toString(op).c_str());
    bool ret = (op.lifetime == OperandLifeTime::CONSTANT_COPY || op.lifetime == OperandLifeTime::CONSTANT_REFERENCE);
    VLOG(L1, "%s", ret ? "Const" : "Non-Const");
    VLOG(L1, "---------------------------------------------");
    return ret;
}


bool PreparedModel::operationAdd(const Operation& operation)
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

bool PreparedModel::operationAveragePool2D(const Operation& operation)
{
  VLOG(L1, "OperationType::AVERAGE_POOL_2D");
  /*
   * * Inputs:
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
  Point2D pad_start = {PARAM_I32(1),PARAM_I32(3)};
  Point2D pad_end = {PARAM_I32(2), PARAM_I32(4)};
  Point2D stride = {PARAM_I32(5),PARAM_I32(6)};
  Point2D kernel = {PARAM_I32(7), PARAM_I32(8)};
  auto out = Pooling(getPort(operation.inputs[0]), kernel, stride, pad_start, pad_end, InferenceEngine::PoolingLayer::PoolType::AVG);
  mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(9));

  return true;
}

bool PreparedModel::operationConCat(const Operation& operation)
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

bool PreparedModel::operationConv2D(const Operation& operation)
{
  VLOG(L1, "OperationType::CONV_2D");
  /*
   * Inputs:
   * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
   *    specifying the filter.
   * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *    For input tensor of {@link OperandType::TENSOR_FLOAT32} type, the bias should
   *    also be of {@link OperandType::TENSOR_FLOAT32}.
   *    For input tensor of {@link OperandType::TENSOR_QUANT8_ASYMM} type, the bias
   *    should be of {@link OperandType::TENSOR_INT32}.
   * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
   * 8: An INT32 value, specifying the output stride in the ‘height’ dimension.
   * 9: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
   *    Specifies the activation to invoke on the result of each addition.                 *
   */

  auto filter = GetConstOperandAsTensor(operation.inputs[1]);
  auto bias = GetConstOperandAsTensor(operation.inputs[2]);


  ConvolutionParams prms;
  prms.weights = static_cast<IRBlob::Ptr>(filter); // permute OHWI to OIHW (0->0, 3->1, 1->2, 2->3)
  const auto dims = prms.weights->getTensorDesc().getDims();
  auto input = getPort(operation.inputs[0]);
  const auto indims = input->getTensorDesc().getDims();
  int  fusion_index = -1;
  if (operation.inputs.size() == 7) {//PAD SAME
      const auto pad_type = PARAM_I32(3);
      int stride_width = PARAM_I32(4);
      int stride_height = PARAM_I32(5);

      int input_height = indims[2];
      int input_width = indims[3];

      int filter_height = dims[2];
      int filter_width = dims[3];
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
          int width_output_size = (input_width + stride_width - 1) / stride_width;
          int width_needed_input = (width_output_size - 1) * stride_width + filter_width;
          int width_total_padding = std::max(0, width_needed_input - width_output_size);

          int height_output_size = (input_height + stride_height - 1) / stride_height;
          int height_needed_input = (height_output_size - 1) * stride_height + filter_height;
          int height_total_padding = std::max(0, height_needed_input - height_output_size);


          int width_padding_to_beginning =  width_total_padding / 2;
          int width_padding_to_end = (width_total_padding + 1)/ 2;
          int height_padding_to_beginning =  height_total_padding / 2;
          int height_padding_to_end = (height_total_padding + 1)/ 2;

          prms.pad_start = {width_padding_to_beginning, height_padding_to_beginning};
          prms.pad_end = {width_padding_to_end, height_padding_to_end};
      } else if (pad_type == kPaddingValid) {
          /**
           * VALID padding.
           * No padding. When the input size is not evenly divisible by
           * the filter size, the input at the end that could not fill
           * the whole filter tile will simply be ignored.
           */
          prms.pad_start = {0, 0};
          prms.pad_end = {0, 0};
      }
      prms.stride = {stride_width, stride_height};
      prms.kernel = {(int)dims[3], (int)dims[2]};
      prms.num_output_planes = dims[0]; //depth out
      fusion_index = 6;
  } else if (operation.inputs.size() == 10) {
      prms.pad_start = {PARAM_I32(3), PARAM_I32(5)};
      prms.pad_end = {PARAM_I32(4), PARAM_I32(6)};
      prms.stride = {PARAM_I32(7), PARAM_I32(8)};
      prms.kernel = {(int)dims[3], (int)dims[2]};
      prms.num_output_planes = dims[0]; //depth out
      fusion_index = 9;
  }

  if (bias->size() != prms.num_output_planes){
      VLOG(L1, "biases size mismatch filer's depth");
      nnAssert(false);
    }

  // input_size (validate)
  if (dims[1] != input->getDims()[1]){
      VLOG(L1, "filter depth_in size mismatch input depth");
      nnAssert(false);
    }
  auto out = Convolution(input, prms) + bias;
  if (fusion_index < 0) {
      VLOG(L1, "invalid fusion index");
      nnAssert(false);
  }
  mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(fusion_index));

  VLOG(L1, "----------------------------------------------");
  VLOGDIMS(L1, indims, "inputs dims");
  VLOGDIMS(L1, dims, "weights dims");
  VLOG(L1, "----------------------------------------------");

  return true;
}

bool PreparedModel::operationDepthwiseConv2D(const Operation& operation)
{
  VLOG(L1, "OperationType::DEPTHWISE_CONV_2D");
  /*
   * Inputs:
   * 0: A 4-D tensor, of shape [batches, height (3), width (3), depth_in (2)], specifying the input.
   * 1: A 4-D tensor, of shape [1, filter_height (2), filter_width (2), depth_out (4)], 1,H,W,(G,I) = H,W,G,I -> G,O,I,H,W (I=1) = G,O,H,W
   *    specifying the filter.
   * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *    For input tensor of {@link OperandType::TENSOR_FLOAT32} type, the bias should
   *    also be of {@link OperandType::TENSOR_FLOAT32}.
   *    For input tensor of {@link OperandType::TENSOR_QUANT8_ASYMM} type, the bias
   *    should be of {@link OperandType::TENSOR_INT32}.
   * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
   * 8: An INT32 value, specifying the output stride in the ‘height’ dimension.
   * 9: An INT32 value, specifying the depthwise multiplier.
   * 10: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
   */
  auto input = getPort(operation.inputs[0]);
  auto filter = GetConstOperandAsTensor(operation.inputs[1]);
  auto bias = GetConstOperandAsTensor(operation.inputs[2]);

  const auto indims = input->getTensorDesc().getDims();


/*
  ConvolutionParams prms;
  // here real weights are not 1,H,W,O since out has groups in it (I=1),H,W,G,O/G, and we use G,(O/G),(I=1),H,W
  const auto fdims = filter->getTensorDesc().getDims();

  for (auto i = 0; i < fdims.size(); i++)
  VLOG(L1, "filter fdims[%d] = %d ", i, fdims[i]);

  auto H = fdims[1];
  auto W = fdims[2];
  auto D = fdims[3];
  auto G = prms.groups = PARAM_I32(9);
  auto I = input->getTensorDesc().getDims()[1]; // IE laypout NCHW
  VLOG(L1, "H = %lu W = %lu  G = %d D = %lu I= %lu D/I = %lu ", H, W, G, D, I, D/I);
  filter->getTensorDesc().reshape({H,W,static_cast<unsigned long>(G),D/I},Layout::ANY);

  prms.weights = static_cast<IRBlob::Ptr>(Permute(filter, {3, 0, 1, 2})); // permute IHWO to OIHW
  const auto dims = prms.weights->getTensorDesc().getDims();
  //call reshape with single vector with size of weights
  prms.weights->getTensorDesc().reshape({filter->size()},Layout::ANY);
*/

  //depthwise: vpu use oihw as shape, format is ihwo. (ihwo->iohw->oihw)
  ConvolutionParams prms;
  prms.weights = static_cast<IRBlob::Ptr>(Permute(filter, {1, 0, 2, 3})); // permute HWOG to GOHW   //3012
  const auto wdims = prms.weights->getTensorDesc().getDims();

  prms.groups = PARAM_I32(9);
  prms.pad_start = {PARAM_I32(3), PARAM_I32(5)};
  prms.pad_end = {PARAM_I32(4), PARAM_I32(6)};
  prms.stride = {PARAM_I32(7), PARAM_I32(8)};
  //prms.kernel = {(int)dims[2], (int)dims[1]};
  prms.kernel = {(int)wdims[3], (int)wdims[2]};
  //prms.num_output_planes = dims[3]; // depth out
  prms.num_output_planes = wdims[0]; // depth out

  uint32_t batches = indims[0];
  uint32_t channels = indims[1];
  uint32_t input_height = indims[2];
  uint32_t input_width = indims[3];

  uint32_t filter_out = wdims[0];
  uint32_t filter_in = wdims[1];
  uint32_t filter_height = wdims[2];
  uint32_t filter_width = wdims[3];

  int32_t padding_left, padding_right;
  int32_t padding_top, padding_bottom;
  int32_t stride_width, stride_height;
  uint32_t depth_multiplier = 0;
  int32_t activation = PARAM_I32(10);

  depth_multiplier = PARAM_I32(9);


  nnAssert(filter_out == channels * depth_multiplier);
  VLOG(L1, "batches %d, channels %d, input_height: %d, input_width %d",
           batches, channels, input_height, input_width);
  VLOG(L1, "channels_in %d, channels_out %d, filter_height: %d, filter_width %d",
           filter_in, filter_out, filter_height, filter_width);
  VLOG(L1, "depth multiplier %d", depth_multiplier);

  //if (bias->size() != prms.num_output_planes*prms.groups){
  if (bias->size() != wdims[0]){
      VLOG(L1, "biases size mismatch filer's depth");
      nnAssert(false);
  }
  //auto input = getPort(operation.inputs[0]);
  // input_size (validate)

  if (prms.groups != input->getDims()[1]){
      VLOG(L1, "input features are not equal depth multiplier");
      nnAssert(false);
    }

  for (auto i = 0; i < wdims.size(); i++)
  VLOG(L1, "weights wdims[%d] = %d ", i, wdims[i]);

/*
  //auto pmem = insertReorderIfNeed(&filter, memory::format::oihw, type_conv_filter);

  filter.shape = {channels, depth_multiplier, 1, filter_height, filter_width};
  //add to stub pmem, then can be pick up later
  addStubPmem(&filter, new memory({{filter.shape, type_conv_filter, format_filter_group},
                                  *cpu_engine}, pmem->get_data_handle()));
*/
  TensorDims newDims = {channels, depth_multiplier, 1, filter_height, filter_width};
  prms.weights->getTensorDesc().setDims(newDims);

  auto out = Convolution(input, prms) + bias;

/*
  int32_t output_height = computeOutSize(input_height, filter_height, stride_height,
                                      padding_top, padding_bottom);
  int32_t output_width = computeOutSize(input_width, filter_width, stride_width,
                                     padding_left, padding_right);
  //get output shape, mkldnn define shape as nchw
  output.shape = {batches, filter_out, output_height, output_width};
*/
  mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(10));

  return true;
}

bool PreparedModel::operationFullyConnected(const Operation& operation)
{
  VLOG(L1, "OperationType::FULLY_CONNECTED");
  /*
   * Inputs:
   * 0: A tensor, specifying the input. If rank is greater than 2, then it gets flattened to
   *    a 2-D Tensor. The 2-D Tensor is handled as if dimensions corresponded to shape
   *    [batch_size, input_size], where “batch_size” corresponds to the batching dimension,
   *    and “input_size” is the size of the input.
   * 1: A 2-D tensor, specifying the weights, of shape [num_units, input_size], where “num_units”
   *    corresponds to the number of output nodes.
   * 2: A 1-D tensor, of shape [num_units], specifying the bias.
   *    For input tensor of {@link OperandType::TENSOR_FLOAT32} type, the bias should
   *    also be of {@link OperandType::TENSOR_FLOAT32}.
   *    For input tensor of {@link OperandType::TENSOR_QUANT8_ASYMM} type, the bias
   *    should be of {@link OperandType::TENSOR_INT32}.
   * 3: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
   *    Specifies the activation to invoke on the result of each addition.
   */
/*
   const hidl_vec<uint32_t>& ins = operation.inputs;
   const hidl_vec<uint32_t>& outs = operation.outputs;

   RunTimeOperandInfo& input   = mOperands[ins[0]];
   RunTimeOperandInfo& weights = mOperands[ins[1]];
   RunTimeOperandInfo& bias    = mOperands[ins[2]];

   int32_t activation = getScalarData<int32_t>(mOperands[ins[3]]);

   RunTimeOperandInfo& output = mOperands[outs[0]];
   Shape outShape = output.shape();

   uint32_t input_size = getNumberOfElements(input);
   uint32_t num_units  = getSizeOfDimension(weights, 0);
   uint32_t batch_size = input_size / getSizeOfDimension(weights, 1);

   NN_OPS_CHECK(getSizeOfDimension(bias, 0) == num_units);
   NN_OPS_CHECK(getSizeOfDimension(weights, 1) * batch_size == input_size);
   NN_OPS_CHECK(getNumberOfDimensions(weights) == 2);

   output->type = input.type;
   output->dimensions = {batch_size, num_units};

*/
    auto weights = GetConstOperandAsTensor(operation.inputs[1]);
    auto bias = GetConstOperandAsTensor(operation.inputs[2]);
    auto input = getPort(operation.inputs[0]);

    auto indims = input->getTensorDesc().getDims();
    for (auto i = 0; i < indims.size(); i++)
    VLOG(L1, "input dims[%d] = %d ", i, indims[i]);

    auto wdims = weights->getTensorDesc().getDims();
    for (auto i = 0; i < wdims.size(); i++)
    VLOG(L1, "weights dims[%d] = %d ", i, wdims[i]);

    //input is [batch_size, input_size], weights is [num_unit, input_size]
    nnAssert(indims[1] == wdims[1]);

    if (input->getDims().size()>2)
    {
        // todo: could be we need to rotate the input weights to reflect the different layout of input tensor
        // when it is not 2D: NHWC vs NCHW in IE
        auto dims = input->getDims();
        input = Reshape({dims[0], product(dims)/dims[0]}, input);
    }

    //FIX ME : Work around since input size indims[0] != output notes (wdims[0])
    auto dims = permuteDims(weights->getTensorDesc().getDims(), {0, 1});
    dims[0] = indims[0];
    weights->getTensorDesc().setDims(dims);
    //WA end

    auto out = weights*input + bias;

    //output->dimensions = {batch_size, num_units};
    //output.shape = {input.shape[0], weights.shape[0]};
    //InferenceEngine::SizeVector outDims = {indims[0], wdims[0]};
    //out->getTensorDesc().setDims(outDims);
    //out->setDims(outDims);

    mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(3));

    return true;
}

bool PreparedModel::operationL2Normalization(const Operation& operation)
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

bool PreparedModel::operationLRN(const Operation& operation)
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

bool PreparedModel::operationMaxPool2D(const Operation& operation)
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
    Point2D pad_start = {PARAM_I32(1),PARAM_I32(3)};
    Point2D pad_end = {PARAM_I32(2), PARAM_I32(4)};
    Point2D stride = {PARAM_I32(5),PARAM_I32(6)};
    Point2D kernel = {PARAM_I32(7), PARAM_I32(8)};
    auto out = Pooling(getPort(operation.inputs[0]), kernel, stride, pad_start, pad_end,
                       InferenceEngine::PoolingLayer::PoolType::MAX);
    mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(9));

    return true;
}

bool PreparedModel::operationLogisticSigmoid(const Operation& operation)
{
  VLOG(L1, "OperationType::LOGISTIC");
  mPorts[operation.outputs[0]] = Sigmoid(getPort(operation.inputs[0]));

  return true;
}
/*
bool PreparedModel::operationLSTM(const Operation& operation)
{
    VLOG("operation type LSTM is supported, but not yet in this implementation");
    nnAssert(true);

    //return true;
}
*/
bool PreparedModel::operationMUL(const Operation& operation)
{
    mPorts[operation.outputs[0]] = handleFusion(getPort(operation.inputs[0])*getPort(operation.inputs[1]), PARAM_I32(2));
    return true;
}

bool PreparedModel::operationRELU(const Operation& operation)
{
    VLOG(L1, "OperationType::RELU");
    mPorts[operation.outputs[0]] = ReLU(getPort(operation.inputs[0]));
    return true;
}

bool PreparedModel::operationRELU1(const Operation& operation)
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

bool PreparedModel::operationRELU6(const Operation& operation)
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

static inline size_t sizeOf(const TensorDims &dims)
{
    size_t ret = dims[0];
    for(int i = 1; i < dims.size(); ++i) ret *= dims[i];
    return ret;
}

bool PreparedModel::operationReshape(const Operation& operation)
{
    VLOG(L1, "OperationType::RESHAPE");
    /*
     * * Inputs:
     * 0: A tensor, specifying the tensor to be reshaped.
     * 1: A 1-D tensor of type {@link OperandType::TENSOR_INT32}, defining the shape
     *    of the output tensor. The number of elements implied by shape must be the same
     *    as the number of elements in the input tensor.
     */
    /* todo: We need to be careful here, inter-tensors are in different order,
     *       could be we need to reflect this also in reshape..
     * */
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

    auto dims = toDims(GetConstVecOperand<uint32_t>(mModel, operation.inputs[1]));
    int index = -1;
    auto shape = 1;
    for (auto i = 0; i < dims.size(); i++) {
        VLOG(L1, "operand1: shape of output tensor dims[%d] = %d ", i, dims[i]);
        if ((int)dims[i] < 0) {
            index = i;
            VLOG(L1, "index = %d", i);
            continue;
        }
        shape *= dims[i];
    }
    if (index >= 0) {
        dims[index] = (uint32_t)(sizeOf((getPort(operation.inputs[0]))->getDims()) / shape);   
        VLOG(L1, "size: %d, index = %d, %d", sizeOf((getPort(operation.inputs[0]))->getDims()),index, dims[index]);
    }

    for (auto i = 0; i < dims.size(); i++)
        VLOG(L1, "operand1: shape of output tensor dims[%d] = %d ", i, dims[i]);
       
    mPorts[operation.outputs[0]] = Reshape(dims, getPort(operation.inputs[0]));

    return true;

}

bool PreparedModel::operationSoftmax(const Operation& operation)
{
    VLOG(L1, "OperationType::SOFTMAX");

    mPorts[operation.outputs[0]] = Softmax(getPort(operation.inputs[0]));
    float scale = PARAM_FP(1);
    if (scale != 1.0f) {
        ALOGE("scale of softmax not suported");
        nnAssert(false);
    }

    return true;
}

bool PreparedModel::operationTANH(const Operation& operation)
{
    VLOG(L1, "OperationType::TANH");
    mPorts[operation.outputs[0]] = Tanh(getPort(operation.inputs[0]));

    return true;
}

void PreparedModel::convertModel(IRDocument &mNet)
{
    for (auto operation : mModel.operations)
    {
        switch (operation.type)
        {
            case OperationType::ADD: {
                VLOG(L1, "OperationType::ADD");
                OutputPort out;
                bool isIn0Const = isConst(operation.inputs[0]);
                bool isIn1Const = isConst(operation.inputs[1]);
                VLOG(L1, "isIn0Const = %d isIn1Const = %d \n", isIn0Const, isIn1Const);
                if (isIn0Const || isIn1Const) {
		    if (isIn0Const && isIn1Const) {
          		ALOGE("adding 2 constants, we can do it now and put const as output");
          		nnAssert(true);
          	    }
          	    // this will use ScaleShift
          	    if (isIn0Const) //if operation.inputs[1] is a Model input
          		out = AddConst(mNet, getPort(operation.inputs[1]),GetConstOperandAsTensor(operation.inputs[0]));
          	    else // isIn1Const is const //operation.inputs[0] is a Model input
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
            } break;

            case OperationType::AVERAGE_POOL_2D: {
                /*
                 * * Inputs:
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
                Point2D pad_start = {PARAM_I32(1),PARAM_I32(3)};
                Point2D pad_end = {PARAM_I32(2), PARAM_I32(4)};
                Point2D stride = {PARAM_I32(5),PARAM_I32(6)};
                Point2D kernel = {PARAM_I32(7), PARAM_I32(8)};
                auto out = Pooling(getPort(operation.inputs[0]), kernel, stride, pad_start, pad_end, InferenceEngine::PoolingLayer::PoolType::AVG);
                mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(9));
            } break;
            case OperationType::CONCATENATION: {
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
            } break;
            case OperationType::CONV_2D: {
                /*
                 * Inputs:
                 * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
                 * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
                 *    specifying the filter.
                 * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
                 *    For input tensor of {@link OperandType::TENSOR_FLOAT32} type, the bias should
                 *    also be of {@link OperandType::TENSOR_FLOAT32}.
                 *    For input tensor of {@link OperandType::TENSOR_QUANT8_ASYMM} type, the bias
                 *    should be of {@link OperandType::TENSOR_INT32}.
                 * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
                 * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
                 * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
                 * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
                 * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
                 * 8: An INT32 value, specifying the output stride in the ‘height’ dimension.
                 * 9: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
                 *    Specifies the activation to invoke on the result of each addition.                 *
                 */

                auto filter = GetConstOperandAsTensor(operation.inputs[1]);
                auto bias = GetConstOperandAsTensor(operation.inputs[2]);
                const auto dims = filter->getTensorDesc().getDims();

                ConvolutionParams prms;
                prms.pad_start = {PARAM_I32(3),PARAM_I32(5)};
                prms.pad_end = {PARAM_I32(4), PARAM_I32(6)};
                prms.stride = {PARAM_I32(7),PARAM_I32(8)};
                prms.kernel = {(int)dims[2], (int)dims[1]};
                //prms.weights = static_cast<IRBlob::Ptr>(Permute(filter, {0, 3, 1, 2})); // permute OHWI to OIHW (0->0, 3->1, 1->2, 2->3);
                prms.weights = static_cast<IRBlob::Ptr>(filter); // permute OHWI to OIHW (0->0, 3->1, 1->2, 2->3);
                prms.num_output_planes = dims[0]; //depth out
                if (bias->size() != prms.num_output_planes){
                    VLOG(L1, "biases size mismatch filer's depth");
                    nnAssert(false);
                }
                auto input = getPort(operation.inputs[0]);
                // input_size (validate)
                if (dims[3] != input->getDims()[1]){
                    VLOG(L1, "filter depth_in size mismatch input depth");
                    nnAssert(false);
                }
                auto out = Convolution(input, prms) + bias;
                mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(9));
            } break;
            case OperationType::DEPTHWISE_CONV_2D: {
                /*
                 * Inputs:
                 * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
                 * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
                 *    specifying the filter.
                 * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
                 *    For input tensor of {@link OperandType::TENSOR_FLOAT32} type, the bias should
                 *    also be of {@link OperandType::TENSOR_FLOAT32}.
                 *    For input tensor of {@link OperandType::TENSOR_QUANT8_ASYMM} type, the bias
                 *    should be of {@link OperandType::TENSOR_INT32}.
                 * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
                 * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
                 * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
                 * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
                 * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
                 * 8: An INT32 value, specifying the output stride in the ‘height’ dimension.
                 * 9: An INT32 value, specifying the depthwise multiplier.
                 * 10: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
                 */
                auto filter = GetConstOperandAsTensor(operation.inputs[1]);
                auto bias = GetConstOperandAsTensor(operation.inputs[2]);
                const auto dims = filter->getTensorDesc().getDims();

                ConvolutionParams prms;
                prms.groups = PARAM_I32(9);
                prms.pad_start = {PARAM_I32(3), PARAM_I32(5)};
                prms.pad_end = {PARAM_I32(4), PARAM_I32(6)};
                prms.stride = {PARAM_I32(7), PARAM_I32(8)};
                prms.kernel = {(int)dims[2], (int)dims[1]};
                // here real weights are not 1,H,W,O since out has groups in it (I=1),H,W,G,O/G, and we use G,(O/G),(I=1),H,W
                prms.weights = static_cast<IRBlob::Ptr>(Permute(filter, {2, 3, 0, 1})); // permute HWGO to GOHW
                prms.num_output_planes = dims[3]; // depth out
                if (bias->size() != prms.num_output_planes){
                    ALOGE("biases size mismatch filer's depth");
                    nnAssert(false);
                }
                auto input = getPort(operation.inputs[0]);
                // input_size (validate)
                if (prms.groups != input->getDims()[1]){
                    ALOGE("input features are not equal depth multiplier");
                    nnAssert(false);
                }
                auto out = Convolution(input, prms) + bias;
                mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(10));
            } break;
            case OperationType::DEPTH_TO_SPACE:
            case OperationType::DEQUANTIZE:
            case OperationType::EMBEDDING_LOOKUP:
            case OperationType::HASHTABLE_LOOKUP:
            case OperationType::FLOOR: {
                ALOGE("operation type = %d is not supported", operation.type);
                nnAssert(false);
            } break;
            case OperationType::FULLY_CONNECTED: {
                /*
                 * Inputs:
                 * 0: A tensor, specifying the input. If rank is greater than 2, then it gets flattened to
                 *    a 2-D Tensor. The 2-D Tensor is handled as if dimensions corresponded to shape
                 *    [batch_size, input_size], where “batch_size” corresponds to the batching dimension,
                 *    and “input_size” is the size of the input.
                 * 1: A 2-D tensor, specifying the weights, of shape [num_units, input_size], where “num_units”
                 *    corresponds to the number of output nodes.
                 * 2: A 1-D tensor, of shape [num_units], specifying the bias.
                 *    For input tensor of {@link OperandType::TENSOR_FLOAT32} type, the bias should
                 *    also be of {@link OperandType::TENSOR_FLOAT32}.
                 *    For input tensor of {@link OperandType::TENSOR_QUANT8_ASYMM} type, the bias
                 *    should be of {@link OperandType::TENSOR_INT32}.
                 * 3: An INT32 value, and has to be one of the {@link FusedActivationFunc} values.
                 *    Specifies the activation to invoke on the result of each addition.
                 */
                auto weights = GetConstOperandAsTensor(operation.inputs[1]);
                auto bias = GetConstOperandAsTensor(operation.inputs[2]);
                auto input = getPort(operation.inputs[0]);
                if (input->getDims().size()>2)
                {
                    // todo: could be we need to rotate the input weights to reflect the different layout of input tensor
                    // when it is not 2D: NHWC vs NCHW in IE
                    auto dims = input->getDims();
                    input = Reshape({dims[0], product(dims)/dims[0]}, input);
                }

                auto out = weights*input + bias;
                mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(3));
            } break;
            case OperationType::L2_NORMALIZATION: {
                mPorts[operation.outputs[0]] = L2Normalization(getPort(operation.inputs[0]), true, false);
            } break;
            case OperationType::L2_POOL_2D: {
                ALOGE("operation type L2_POOL_2D is not supported");
                nnAssert(false);
            } break;
            case OperationType::LOCAL_RESPONSE_NORMALIZATION: {
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
                mPorts[operation.outputs[0]] = LRN(getPort(operation.inputs[0]), alpha, beta, size, true, k);
            } break;
            case OperationType::LOGISTIC: {
                mPorts[operation.outputs[0]] = Sigmoid(getPort(operation.inputs[0]));
            } break;
            case OperationType::LSH_PROJECTION: {
                VLOG(L1, "operation type LSH_PROJECTION is not supported");
                nnAssert(false);
            } break;
            case OperationType::LSTM: {
                VLOG(L1, "operation type LSTM is supported, but not yet in this implementation");
                nnAssert(false);
            } break;
            case OperationType::MAX_POOL_2D: {
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
                Point2D pad_start = {PARAM_I32(1),PARAM_I32(3)};
                Point2D pad_end = {PARAM_I32(2), PARAM_I32(4)};
                Point2D stride = {PARAM_I32(5),PARAM_I32(6)};
                Point2D kernel = {PARAM_I32(7), PARAM_I32(8)};
                auto out = Pooling(getPort(operation.inputs[0]), kernel, stride, pad_start, pad_end,
                                   InferenceEngine::PoolingLayer::PoolType::MAX);
                mPorts[operation.outputs[0]] = handleFusion(out, PARAM_I32(9));
            } break;
            case OperationType::MUL:
                mPorts[operation.outputs[0]] = handleFusion(getPort(operation.inputs[0])*getPort(operation.inputs[1]), PARAM_I32(2));
                break;
            case OperationType::RELU: {
              VLOG(L1, "OperationType::RELU");
              mPorts[operation.outputs[0]] = ReLU(getPort(operation.inputs[0]));
            } break;

            case OperationType::RELU1: {
              VLOG(L1, "OperationType::RELU1");
              mPorts[operation.outputs[0]] = Clamp(getPort(operation.inputs[0]),-1,1);
            } break;
            case OperationType::RELU6: {
              VLOG(L1, "OperationType::RELU6");
              mPorts[operation.outputs[0]] = Clamp(getPort(operation.inputs[0]),0,6);
            } break;
            case OperationType::RESHAPE: {
                /*
                 * * Inputs:
                 * 0: A tensor, specifying the tensor to be reshaped.
                 * 1: A 1-D tensor of type {@link OperandType::TENSOR_INT32}, defining the shape
                 *    of the output tensor. The number of elements implied by shape must be the same
                 *    as the number of elements in the input tensor.
                 */
                /* todo: We need to be careful here, inter-tensors are in different order,
                 *       could be we need to reflect this also in reshape..
                 * */
                auto dims = toDims(GetConstVecOperand<uint32_t>(mModel, operation.inputs[1]));
                mPorts[operation.outputs[0]] = Reshape(dims, getPort(operation.inputs[0]));
            } break;
            case OperationType::RESIZE_BILINEAR:
            case OperationType::RNN:
            case OperationType::SOFTMAX: {
                mPorts[operation.outputs[0]] = Softmax(getPort(operation.inputs[0]));
                float scale = PARAM_FP(1);
                if (scale != 1.0f) {
                    ALOGE("scale of softmax not suported");
                    nnAssert(false);
                  }
            } break;
            case OperationType::SPACE_TO_DEPTH:
            case OperationType::SVDF: {
                ALOGE("operation type = %d is not supported", operation.type);
                nnAssert(false);
            } break;
            case OperationType::TANH: {
              VLOG(L1, "OperationType::TANH");
              mPorts[operation.outputs[0]] = Tanh(getPort(operation.inputs[0]));
            } break;
            default: {
                ALOGE("operation type = %d is unknown", operation.type);
                nnAssert(false);
            }
        }
    }
	// loop over all outputs
	for (auto i : mModel.outputIndexes)
	{
//		mPorts[i]->setLayout(NHWC);
/*
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
*/
		mPorts[i]->setPrecision(InferenceEngine::Precision::FP32);
		mNet.addOutput(mPorts[i]);

    VLOG(L1, "intialization for output data mPorts[%d]->name = %s\n", i, mPorts[i]->name.c_str());
	}
}

void PreparedModel::initializeInput(RunTimeOperandInfo* input)
{
}
//from {pmem, shape} to {new pmem, type, buffer, format, length}
void PreparedModel::finalizeOutput(/*RunTimeOperandInfo* output */)
{
  VLOG(L1, "finalize Output");
  for (auto i : mModel.outputIndexes)
  {
    int dims_size = mOperands[i].dimensions.size();

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

    //mPorts[i]->setPrecision(InferenceEngine::Precision::FP16);
    mPorts[i]->setPrecision(InferenceEngine::Precision::FP32);
    mNet.addOutput(mPorts[i]);

    VLOG(L1, "mPorts[%d] %s dims size %d", i, mPorts[i]->name.c_str(), dims_size);
    VLOGDIMS(L1, mOperands[i].dimensions, "Output dims:");
    VLOGDIMS(L1, mPorts[i]->getDims(), "Real Output dims:");

    auto dims = mPorts[i]->getDims();
    for (auto j = 0; j < dims.size(); j++)
    VLOG(L1, "output dims[%d] = %d & set output dims[%d] = %d ", j, mOperands[i].dimensions[j], j, dims[j]);
    VLOG(L1, "intialization for output data mPorts[%d]->name = %s\n", i, mPorts[i]->name.c_str());
  }
}

IRBlob::Ptr VpuPreparedModel::GetConstOperandAsTensor(uint32_t index)
{
    //const auto op = model.operands.at(index);
    const auto op = mModel.operands[index];
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, index, len);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
#ifndef MYRIAD_FP32  //Myriad only supprts FP16
            vec<unsigned int> order;
            if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
            else if (op.dimensions.size() == 2) order = {0, 1};
            else order = {0}; //(op.dimensions.size() < 2)
            TensorDesc td(InferenceEngine::Precision::FP16, permuteDims(toDims(op.dimensions), order), Layout::ANY);
            // todo: create a readOnly blob that accepts const pointers
            InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
            blob->allocate();
            auto mem = blob->data();
            short *fp16Array = mem.as<short*>();
            // convert from [(float *)buf, len] to fp16Array,
            uint32_t nelem = getNumberOfElements(op.dimensions);
            VLOG(L1, "Model buffer oplength = %d bytes nelem= %d fp16Array= %d bytes sizeof model buf= %d bytes\n", len , nelem, sizeof(fp16Array), sizeof(buf));
            if (blob->size() != nelem) {
                VLOG(L1, "Model buffer len = %d bytes nelem= %d fp16Array= %d bytes\n",len , nelem, sizeof(fp16Array));
                nnAssert(false);
            }
            f32tof16Arrays(fp16Array, (float *)buf, nelem);
            return blob;
#else //FP32 support
            vec<unsigned int> order;
            if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
            else if (op.dimensions.size() == 2) order = {0, 1};
            else order = {0}; //(op.dimensions.size() < 2)
            TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), order), Layout::ANY);
            // todo: create a readOnly blob that accepts const pointers
            //return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
#endif
    } else if (op.type == OperandType::TENSOR_INT32) {

        VLOG(L1, "check if const tensors of type IN32 supported");
        //nnAssert(true);
        TensorDesc td(InferenceEngine::Precision::I32, toDims(op.dimensions), Layout::ANY);
//        return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
    } else {
        VLOG(L1, "not supporting const tensors of type ", op.type);
        nnAssert(false);
    }
    return nullptr;
}

Blob::Ptr VpuPreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
    //const auto op = model.operands[index];
    //uint32_t len;
    //const uint8_t *buf = GetOperandMemory(model, index, len);

    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {

#ifndef MYRIAD_FP16  //Myriad supports FP32 only for network input/output
    if (op.lifetime == OperandLifeTime::MODEL_INPUT) {

      //        TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), Layout::ANY);
              // todo: create a readOnly blob that accepts const pointers
      //        return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
      vec<unsigned int> order;
      if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
      else if (op.dimensions.size() == 2) order = {0, 1};
      else order = {0}; //(op.dimensions.size() < 2)

      TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), order), Layout::ANY);
          // todo: create a readOnly blob that accepts const pointers
  		//InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
      InferenceEngine::TBlob<float>::Ptr blob = InferenceEngine::make_shared_blob<float>(td, (float *)buf, len);
  		//blob->allocate();
    VLOG(L1, "Create input blob");
    return blob;
		}
    else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
//      TensorDesc td(InferenceEngine::Precision::FP16, toDims(op.dimensions), Layout::ANY);
      // todo: create a readOnly blob that accepts const pointers
//      return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
      TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), Layout::ANY); //nhwc
      //TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), {0,3,1,2}), Layout::ANY);  //nhwc->nchw
          // todo: create a readOnly blob that accepts const pointers
      InferenceEngine::TBlob<float>::Ptr blob = InferenceEngine::make_shared_blob<float>(td, (float *)buf, len);
      VLOG(L1, "Create output blob");
      return blob;
    }

#else //FP16 support if Myriad does not support FP32 for network input/output

    if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
      TensorDesc td(InferenceEngine::Precision::FP16, toDims(op.dimensions), Layout::ANY);
          // todo: create a readOnly blob that accepts const pointers
      //InferenceEngine::TBlob<short>::Ptr blob = std::make_shared<InferenceEngine::TBlob<short>>(td);
      vec<unsigned int> order;
      if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
      else if (op.dimensions.size() == 2) order = {0, 1};
      else order = {0}; //(op.dimensions.size() < 2)

      InferenceEngine::TBlob<short>::Ptr blob = InferenceEngine::make_shared_blob<short, InferenceEngine::SizeVector>(InferenceEngine::Precision::FP16, permuteDims(toDims(op.dimensions), order));
      //Blob::Ptr Blob::CreateFromData(const DataPtr &data)
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
    //      TensorDesc td(InferenceEngine::Precision::FP16, toDims(op.dimensions), Layout::ANY);
      // todo: create a readOnly blob that accepts const pointers
    //      return std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
      TensorDesc td(InferenceEngine::Precision::FP16, toDims(op.dimensions), Layout::ANY);
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

IRBlob::Ptr CpuPreparedModel::GetConstOperandAsTensor(uint32_t index)
{
    dumpOperand(index);
    const auto op = mModel.operands[index];
    uint32_t len;
    const uint8_t *buf = GetOperandMemory(mModel, index, len);
    VLOG(L1, "CpuPreparedModel:: Operand: index: %d, len: %d, buf: %p", index, len, buf);
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        vec<unsigned int> order;
        if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
        else if (op.dimensions.size() == 2) order = {0, 1};
        else order = {0}; //(op.dimensions.size() < 2)
        TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), order), Layout::ANY);
        if (buf == nullptr)
            VLOG(L1, "TENSOR_FLOAT32 buf is NULL !!!!!!!!!!!!!!!");
        InferenceEngine::TBlob<float>::Ptr blob = std::make_shared<InferenceEngine::TBlob<float>>(td, (float *)buf, len);
        return blob;
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

Blob::Ptr CpuPreparedModel::GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t *buf, uint32_t& len)
{
    if (op.type == OperandType::TENSOR_FLOAT32 || op.type == OperandType::FLOAT32) {
        if (op.lifetime == OperandLifeTime::MODEL_INPUT) {
            vec<unsigned int> order;
            if (op.dimensions.size() == 4) order = {0,3,1,2};  //nhwc -> nchw
            else if (op.dimensions.size() == 2) order = {0, 1};
            else order = {0}; //(op.dimensions.size() < 2)

            VLOG(L1, "Create input blob !!!!");
            TensorDesc td(InferenceEngine::Precision::FP32, permuteDims(toDims(op.dimensions), order), Layout::ANY);
            if (buf == nullptr)
                VLOG(L1, "MODEL_INPUT buf is NULL !!!!!!!!!!!!!!!");
            InferenceEngine::TBlob<float>::Ptr blob = InferenceEngine::make_shared_blob<float>(td, (float *)buf, len);
            return blob;
        } else if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
            VLOG(L1, "Create output blob !!!!");
            TensorDesc td(InferenceEngine::Precision::FP32, toDims(op.dimensions), Layout::ANY); //nhwc
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

}  // namespace driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
