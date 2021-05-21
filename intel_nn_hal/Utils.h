#ifndef ANDROID_ML_NN_UTILS_H
#define ANDROID_ML_NN_UTILS_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hardware/neuralnetworks/1.3/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>
#include <utility>
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <NeuralNetworks.h>
#include <thread>
#include <cpuid.h>

#include "ValidateHal.h"
#include "IRBuilder.h"
#include "Driver.h"

#define PERF_COUNTERS
//#define CACHING
//#define NN_DEBUG

enum DebugLevel {
    L0,
    L1,
    L2,
    L3,
    L4,
};

#define DECODER_TOKEN_STR "Decoder"
#define ENC0_TOKEN_STR "Encoder0"
#define ENC1_TOKEN_STR "Encoder1"

extern unsigned int debugMask;
//unsigned int debugMask = ((1 << (L1 + 1)) - 1);
//TODO: Fix this

#ifdef NN_DEBUG
#define VLOG(l, x, ...)                                                          \
    do {                                                                         \
         if (debugMask & (1 << l)) ALOGI("[%s] " x, __FUNCTION__, ##__VA_ARGS__); \
    } while (0)

#define VLOGDIMS(l, d, header)                                                         \
    do {                                                                               \
        auto size = (d).size();                                                        \
        VLOG(l, "%s: vectors {%d, %d, %d, %d}", header, (d)[0], size > 1 ? (d)[1] : 0, \
             size > 2 ? (d)[2] : 0, size > 3 ? (d)[3] : 0);                            \
    } while (0)

#define dumpOperand(index)                                      \
    do {                                                        \
        const auto op = mModel.main.operands[index];                 \
        ALOGE("---------------------------------------------"); \
        ALOGE("Operand index: %d", index);                      \
        ALOGE("%s", toString(op).c_str());                      \
        ALOGE("---------------------------------------------"); \
    } while (0)

#define dumpOperation(operation)                                \
    do {                                                        \
        ALOGE("---------------------------------------------"); \
        ALOGE("Operation:");                                    \
        ALOGE("%s", toString(operation).c_str());               \
        ALOGE("---------------------------------------------"); \
    } while (0)

#define dumpOperationSupport(operation, support)                    \
    do {                                                            \
        ALOGI("---------------------------------------------");     \
        ALOGI("Operation support: %s", support ? "True" : "False"); \
        ALOGI("%s", toString(operation).c_str());                   \
        ALOGI("---------------------------------------------");     \
    } while (0)

#define dumpOperationParam(operation)             \
    do {                                          \
        ALOGI("dumping operation-params");        \
        ALOGI("%s", toString(operation).c_str()); \
    } while (0)

#else
#define VLOG(...)
#define VLOGDIMS(l, d, header)
#define dumpOperand(...)
#define dumpOperation(operation)
#define dumpOperationSupport(operation, support)
#define dumpOperationParam(operation)
#endif

#define WRONG_DIM (-1)

#define nnAssert(v)                                                                            \
    do {                                                                                       \
        if (!(v)) {                                                                            \
            LOG(ERROR) << "nnAssert failed at " << __FILE__ << ":" << __LINE__ << " - '" << #v \
                       << "'\n";                                                               \
            abort();                                                                           \
        }                                                                                      \
    } while (0)

#define EXPL_PAD_PARAMS_CONV 10
#define IMPL_PAD_PARAMS_CONV 7
#define EXPL_PAD_PARAMS_DW_CONV 11
#define IMPL_PAD_PARAMS_DW_CONV 8
#define SOFTMAX_INPUT_PARAMS 2
#define NHWC_DIM_NUM 4
#define NHWC_CH_IDX 3
#define NHWC_HT_IDX 1
#define NHWC_WD_IDX 2
// operand index as from  1.1/type.hal
#define OP_INPUT_IDX_CONV 0
#define OP_FILTER_IDX_CONV 1
#define OP_BIAS_IDX_CONV 2
#define OP_PADSCHEME_IDX_CONV 3
#define OP_PADL_IDX_CONV 3
#define OP_PADR_IDX_CONV 4
#define OP_PADH_IDX_CONV 5
#define OP_PADW_IDX_CONV 6
#define OP_STRD_WD_IDX_EXPL_CONV 7
#define OP_STRD_HT_IDX_EXPL_CONV 8
#define OP_STRD_WD_IDX_IMPL_CONV 4
#define OP_STRD_HT_IDX_IMPL_CONV 5
#define OP_ACTV_FUNC_IDX_IMPL_CONV 6
#define OP_ACTV_FUNC_IDX_EXPL_CONV 9
#define OP_ACTV_FUNC_IDX_IMPL_DW_CONV 7
#define OP_ACTV_FUNC_IDX_EXPL_DW_CONV 10
#define OP_DW_CONV_DPM_IMPL 6  // depth multiplier
#define OP_DW_CONV_DPM_EXPL 9
#define OP_ADD_OPR1_IDX 0
#define OP_ADD_OPR2_IDX 1

// average_pooling_2d as in type.hal
#define EXPL_PAD_PARAMS_POOL 10
#define IMPL_PAD_PARAMS_POOL 7
#define OP_INPUT_IDX_POOL 0
#define OP_PADL_IDX_POOL 1
#define OP_PADR_IDX_POOL 2
#define OP_PADH_IDX_POOL 3
#define OP_PADW_IDX_POOL 4
#define OP_STRD_WD_IDX_EXPL_POOL 5
#define OP_STRD_HT_IDX_EXPL_POOL 6
#define OP_FLT_WD_IDX_EXPL_POOL 7
#define OP_FLT_HT_IDX_EXPL_POOL 8
#define OP_ACTV_FUNC_IDX_EXPL_POOL 9

#define OP_PADSCHEME_IDX_POOL 1
#define OP_STRD_WD_IDX_IMPL_POOL 2
#define OP_STRD_HT_IDX_IMPL_POOL 3
#define OP_FLT_WD_IDX_IMPL_POOL 4
#define OP_FLT_HT_IDX_IMPL_POOL 5
#define OP_ACTV_FUNC_IDX_IMPL_POOL 6

// fully_connected as in type.hal
#define OP_INPUT_IDX_FC 0
#define OP_WGHT_IDX_FC 1
#define OP_BIAS_IDX_FC 2
#define OP_ACTV_IDX_FC 3
#define FC_INPUT_PARAMS 4

// ADD operation
#define ADD_INPUT_PARAMS 3
#define OP_INPUT0_IDX_ADD 0
#define OP_INPUT1_IDX_ADD 1
#define OP_ACTV_IDX_ADD 2

#define CHECK_OPERAND_2D(params, idx_x, idx_y)                                                 \
    do {                                                                                       \
        VLOG(L1, "As found in %s", __func__);                                                  \
        if (params.x < 0 || params.y < 0) {                                                    \
            VLOG(L1, "Invalid Point2D Operands at index [%d ,%d] , aborting!!", idx_x, idx_y); \
            return false;                                                                      \
        }                                                                                      \
    } while (0)


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class BaseOp {
    std::vector<uint32_t> mOutputIndices;
    std::vector<uint32_t> mInputIndices;

    public:
        virtual bool isCpuOp() = 0;

        virtual bool run() {
            return false;
        };

        virtual std::string getLayerName() { return "";}
        virtual ~BaseOp() {}

        //TODO: Use Blob::Ptr to make it generic
        virtual bool setInputData(uint32_t graph_index, void* dataPtr, uint32_t len)
        {
            return false;
        }

        virtual bool setInputIndex(uint32_t graph_index, uint32_t op_index) {
            return false;
        };

        virtual std::tuple<void*, int32_t> getOutputData() {
            return std::make_pair(nullptr, 0);
        }

        virtual void setOutputIndices(std::vector<uint32_t> indices) {
            for (auto in: indices)
                mOutputIndices.push_back(in);
        }

        virtual std::vector<uint32_t> getOutputIndices() {
            return mOutputIndices;
        }

        virtual void setInputIndices(std::vector<uint32_t> indices) {
            for (auto in: indices)
                mInputIndices.push_back(in);
        }

        virtual std::vector<uint32_t> getInputIndices() {
            return mInputIndices;
        }

        virtual void cleanup() {}
};

class OpContainer {
    std::vector<BaseOp*> opsVec;
    float* output = nullptr;
    bool targetCpu = false;

    public:

        OpContainer(bool cpu) : targetCpu(cpu) {}

        void addOperation(BaseOp* op) {
            opsVec.push_back(op);
        }

        bool run() {
            std::tuple<void*, uint32_t> intermediate_inp;
            int i;
            BaseOp* op;
            if (opsVec.size() == 1) {
                op = opsVec[0];
                op->run();
            }
            else {
            for (i = 0; i < opsVec.size() - 1 ; i++) {
                op = opsVec[i];
                if ( i == 0) {
                    op->run();
                }
                else {
                    std::vector<uint32_t> ip_indices = op->getInputIndices();
                    op->setInputData(ip_indices[0], std::get<0>(intermediate_inp), std::get<1>(intermediate_inp));
                    op->run();
                }
                intermediate_inp = op->getOutputData();
                auto inp_ptr = static_cast<float*>(std::get<0>(intermediate_inp));
            }
            if (i == opsVec.size() - 1) {
                op = opsVec[i];
                std::vector<uint32_t> ip_indices = op->getInputIndices();

                op->setInputData(ip_indices[0], std::get<0>(intermediate_inp), std::get<1>(intermediate_inp));

                op->run();
            }
            }

            return true;
        }

        std::vector<uint32_t> getOutputIndices() {
            std::vector<uint32_t> indices;
            for (auto op: opsVec) {
                auto indexVec = op->getOutputIndices();

                for (auto index: indexVec)
                    indices.push_back(index);
            }

            return indices;
        }

        std::vector<uint32_t> getInputIndices() {
            std::vector<uint32_t> indices;
            for (auto op: opsVec) {
                auto indexVec = op->getInputIndices();

                for (auto index: indexVec)
                    indices.push_back(index);
            }

            return indices;
        }

        BaseOp* getCpuOpFromLayerName(std::string layer) {
            for (auto op: opsVec) {
                if (layer.compare(op->getLayerName()) == 0) {
                    return op;
                }
            }

            return nullptr;
        }

        virtual void cleanup() {
            for (auto op: opsVec) {
                op->cleanup();
            }
        }

        bool isCpuGraph() { return targetCpu; }

};

template <class T>
using vec = std::vector<T>;
typedef uint8_t* memory;

using namespace android::nn;
using namespace IRBuilder;

// The type and dimensions of an operand.
struct Shape {
    OperandType type;
    std::vector<uint32_t> dimensions;
    float scale;
    int32_t offset;
};

enum class DataType {
    UINT32,
    FLOAT32,
    INT8,
    UINT8,
    UINT16,
    INT16
};

enum class DeviceType {
    CPU,
    GNA,
    None
};

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo {
    // std::string name;
    // uint32_t opIdx;
    // void * opIdx;

    // TODO Storing the type here is redundant, as it won't change during execution.
    OperandType type;
    // The type and dimensions of the operand.  The dimensions can
    // change at runtime.  We include the type because it's useful
    // to pass together with the dimension to the functions implementing
    // the operators.
    std::vector<uint32_t> dimensions;

    float scale;
    int32_t zeroPoint;
    // Where the operand's data is stored.  Check the corresponding
    // location information in the model to figure out if this points
    // to memory we have allocated for an temporary operand.
    uint8_t* buffer;
    // The length of the buffer.
    uint32_t length;
    // Whether this is a temporary variable, a model input, a constant, etc.
    V1_3_OperandLifeTime lifetime;
    // Keeps track of how many operations have yet to make use
    // of this temporary variable.  When the count is decremented to 0,
    // we free the buffer.  For non-temporary variables, this count is
    // always 0.
    uint32_t numberOfUsesLeft;
    DataType inputDataType;
    DataType outDataType;

    Shape shape() const {
        return Shape{.type = type, .dimensions = dimensions, .scale = scale, .offset = zeroPoint};
    }
};

// Used to keep a pointer to each of the memory pools.
struct RunTimePoolInfo {
    sp<::android::hidl::memory::V1_0::IMemory> memory;
    hidl_memory hidlMemory;
    uint8_t* buffer = nullptr;

    bool set(const hidl_memory& hidlMemory);
    bool update();
    bool unmap_mem();
};

bool setRunTimePoolInfosFromHidlMemories(std::vector<RunTimePoolInfo>* poolInfos,
                                         const hidl_vec<hidl_memory>& pools);

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

void calculateExplicitPadding(int32_t in_size, int32_t stride, int32_t filter_size,
                              int32_t padding_implicit, int32_t* padding_head,
                              int32_t* padding_tail);

int32_t computeOutSize(int32_t imageSize, int32_t filterSize, int32_t stride, int32_t paddingHead,
                       int32_t paddingTail);

// shape is nchw, dims depends on layout
TensorDims dimsToShape(const std::vector<uint32_t>& dims, InferenceEngine::Layout layout);

// shape is nchw, dims depends on format
std::vector<uint32_t>& shapeToDims(const TensorDims& shape, InferenceEngine::Layout layout);

unsigned short float2half(unsigned f);

size_t sizeOfTensor(const TensorDims& dims);

void floattofp16(short* dst, float* src, unsigned nelem);

// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16 0x7C00U

// small helper function to represent uint32_t value as float32
float asfloat(uint32_t v);

// Function to convert F32 into F16
float f16tof32(short x);

// This function convert f32 to f16 with rounding to nearest value to minimize error
// the denormal values are converted to 0.
short f32tof16(float x);

void f16tof32Arrays(float* dst, const short* src, uint32_t& nelem, float scale = 1,
                    float bias = 0);

void f32tof16Arrays(short* dst, const float* src, uint32_t& nelem, float scale = 1,
                    float bias = 0);

int sizeOfData(OperandType type, std::vector<uint32_t> dims);

size_t getSizeFromInts(int lower, int higher);

uint32_t getNumberOfElements(const vec<uint32_t>& dims);

TensorDims toDims(const vec<uint32_t>& dims);

TensorDims permuteDims(const TensorDims& src, const vec<unsigned int>& order);

// IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int> &order)
IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int>& order);

#define PARAM_I32(i) ParseOperationInput<int32_t>(mModel, operation, i)
#define PARAM_FP(i) ParseOperationInput<float>(mModel, operation, i)

template <typename T>
struct printHelper {
    static void print(const T& value, const char* Obj) {}
};

template <>
struct printHelper<int32_t> {
    static void print(const int32_t& value, const char* operand) {
        VLOG(L1, "Operand: value: %d, %s", value, operand);
    }
};

template <>
struct printHelper<float> {
    static void print(const float& value, const char* operand) {
        VLOG(L1, "Operand: value: %f, %s", value, operand);
    }
};

void writeBufferToFile(std::string filename,
                        const float* buf,
                        size_t length);

#ifdef PERF_COUNTERS
inline void native_cpuid(unsigned int *eax, unsigned int *ebx,
                         unsigned int *ecx, unsigned int *edx) {
    size_t level = *eax;
#if defined(_WIN32) || defined(WIN32)
    int regs[4] = {static_cast<int>(*eax), static_cast<int>(*ebx), static_cast<int>(*ecx), static_cast<int>(*edx)};
    __cpuid(regs, level);
    *eax = static_cast<uint32_t>(regs[0]);
    *ebx = static_cast<uint32_t>(regs[1]);
    *ecx = static_cast<uint32_t>(regs[2]);
    *edx = static_cast<uint32_t>(regs[3]);
#else
    __get_cpuid(level, eax, ebx, ecx, edx);
#endif
}

// return GNA module frequency in MHz
float getGnaFrequencyMHz();

typedef struct _metrics{
    double deQuant_time;
    double quant_time;
    double nw_load_time;
    double avg_infer_time;
    uint64_t infer_calls;
    double irBuild_time;
    std::vector<double> infer_time;

    void reset() {
        deQuant_time = 0.0;
        quant_time = 0.0;
        nw_load_time = 0.0;
        avg_infer_time = 0.0;
        infer_calls = 0;
        irBuild_time =0.0;
        infer_time = {};
    }

    void print() {

        auto avg_infer = 0;
        std::stringstream outputlog;

        if (!infer_time.empty()) {
            avg_infer = std::accumulate(infer_time.begin(), infer_time.end(), 0) / (infer_time.size());

            outputlog << "All infer times: ";
            for (auto t : infer_time) {
                outputlog << t << " ";
            }
            outputlog << std::endl;
        }

        std::cout << std::setw(25) << " Dequant time(ms):"
                  << std::setw(25)   << " Quant time(us):"
                  << std::setw(25)   << " NW load time(ms):"
                  << std::setw(25)   << " Avg infer time(ms):"
                  << std::setw(25)   << " Infer calls:"
                  << std::setw(25)   << " OV IR Build time(ms):"
                  << std::endl;
        std::cout << std::setw(25) <<  deQuant_time
                   << std::setw(25) <<  quant_time
                   << std::setw(25)  << nw_load_time
                   << std::setw(25)  << avg_infer
                   << std::setw(25)  << infer_calls
                   << std::setw(25)  << irBuild_time
                   << std::endl;
        std::cout << "Infer times: " << outputlog.str().c_str();
    }
} metrics;
#endif

std::string getTokenString(const HidlToken& token);
std::string computeHashFromFd(int fd);
}
}
}
}
#endif
