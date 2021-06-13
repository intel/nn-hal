#include "Utils.h"

#include "Driver.h"
#include "IENetwork.h"
#include "BuilderNetwork.h"

#include <fstream>
#include <sys/stat.h>
#include "openssl/evp.h"

unsigned int debugMask = ((1 << (L1 + 1)) - 1);

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

using namespace android::nn;

void calculateExplicitPadding(int32_t in_size, int32_t stride, int32_t filter_size,
                              int32_t padding_implicit, int32_t* padding_head,
                              int32_t* padding_tail) {
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

int32_t computeOutSize(int32_t imageSize, int32_t filterSize, int32_t stride, int32_t paddingHead,
                       int32_t paddingTail) {
    return (imageSize - filterSize + stride + paddingHead + paddingTail) / stride;
}

// shape is nchw, dims depends on layout
TensorDims dimsToShape(const std::vector<uint32_t>& dims, InferenceEngine::Layout layout) {
    VLOG(L3, "layout: %d", static_cast<int>(layout));
    VLOGDIMS(L3, dims, "dims");
    TensorDims shape;
    uint32_t n, c, h, w;
    // 4-D
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

// shape is nchw, dims depends on format
std::vector<uint32_t>& shapeToDims(const TensorDims& shape, InferenceEngine::Layout layout) {
    VLOG(L3, "layout: %d", static_cast<int>(layout));
    VLOGDIMS(L3, shape, "shape");
    uint32_t n, c, h, w;
    std::vector<uint32_t> dims;
    // 1-D
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

    // 4-D
    // vpu accept nchw or oihw.
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
        default:
            VLOG(L1, "unsupported layout %d", layout);
    }

    VLOGDIMS(L3, dims, "dims");
    return dims;
}

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

// Function to convert F32 into F16
// F32: exp_bias:127 SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM.
// F16: exp_bias:15  SEEEEEMM MMMMMMMM
#define EXP_MASK_F32 0x7F800000U
#define EXP_MASK_F16 0x7C00U

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

void f16tof32Arrays(float* dst, const short* src, uint32_t& nelem, float scale,
                    float bias) {
    VLOG(L1, "convert f16tof32Arrays...\n");
    const short* _src = reinterpret_cast<const short*>(src);

    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f16tof32(_src[i]) * scale + bias;
    }
}

void f32tof16Arrays(short* dst, const float* src, uint32_t& nelem, float scale,
                    float bias) {
    VLOG(L1, "convert f32tof16Arrays...");
    for (uint32_t i = 0; i < nelem; i++) {
        dst[i] = f32tof16(src[i] * scale + bias);
    }
}

size_t sizeOfTensor(const TensorDims& dims) {
    size_t ret = dims[0];
    for (int i = 1; i < dims.size(); ++i) ret *= dims[i];
    return ret;
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
            size = 4;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::TENSOR_QUANT8_SYMM:
        case OperandType::TENSOR_QUANT8_ASYMM_SIGNED:
            size = 1;
            break;

        case OperandType::TENSOR_QUANT16_ASYMM:
        case OperandType::TENSOR_QUANT16_SYMM:
            size = 2;
            break;
        default:
            size = 0;
    }
    for (auto d : dims) size *= d;

    return size;
}

size_t getSizeFromInts(int lower, int higher) {
    return (uint32_t)(lower) + ((uint64_t)(uint32_t)(higher) << 32);
}

/* template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
} */

uint32_t getNumberOfElements(const vec<uint32_t>& dims) {
    uint32_t count = 1;
    for (size_t i = 0; i < dims.size(); i++) {
        count *= dims[i];
    }
    return count;
}

TensorDims toDims(const vec<uint32_t>& dims) {
    TensorDims td;
    for (auto d : dims) td.push_back(d);
    return td;
}
/*
template <typename T>
size_t product(const vec<T>& dims) {
    size_t rc = 1;
    for (auto d : dims) rc *= d;
    return rc;
} */

TensorDims permuteDims(const TensorDims& src, const vec<unsigned int>& order) {
    TensorDims ret;
    for (int i = 0; i < src.size(); i++) {
        ret.push_back(src[order[i]]);
    }
    return ret;
}

// IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int> &order)
IRBlob::Ptr Permute(IRBlob::Ptr ptr, const vec<unsigned int>& order) {
    VLOG(L1, "Permute");
    auto orig_dims = ptr->getTensorDesc().getDims();
    auto dims = permuteDims(orig_dims, order);
    ptr->getTensorDesc().setDims(dims);

    return ptr;
}

#define PARAM_I32(i) ParseOperationInput<int32_t>(mModel, operation, i)
#define PARAM_FP(i) ParseOperationInput<float>(mModel, operation, i)
/*
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
}; */

void createDirs(std::string path) {
    char delim = '/';
    int start = 0;

    auto pos = path.find(delim);
    while (pos != std::string::npos) {
		auto dir = path.substr(start, pos - start+1);

        struct stat sb;
        if (!((stat(dir.c_str(), &sb) == 0) && (S_ISDIR(sb.st_mode)))) {
            if (mkdir(dir.c_str(), 0777) != 0)
                std::cout << "failed to create folder: " << dir << std::endl;
		}
		pos = path.find(delim, pos+1);
	}
}

void writeBufferToFile(std::string filename,
                        const float* buf,
                        size_t length) {
	createDirs(filename);

    std::ofstream ofs;
    ofs.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
    for (auto i =0; i < length; i++) {
        ofs << buf[i] << "\n";
    }
    ofs.close();
}

#ifdef CACHING
std::string getTokenString(const HidlToken& token) {
    std::string tokenStr(ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN * 2 + 1, '0');
    for (uint32_t i = 0; i < ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN; i++) {
        tokenStr[i * 2] = 'A' + (token[i] & 0x0F);
        tokenStr[i * 2 + 1] = 'A' + (token[i] >> 4);
    }
    return  tokenStr;
}

std::string computeHashFromFd(int modelFd) {
    unsigned char hashValue[EVP_MAX_MD_SIZE];
    unsigned int hashLen;

    OpenSSL_add_all_algorithms();
    auto md = EVP_get_digestbyname("SHA512");
    if (md != NULL) {
        EVP_MD_CTX *mdctx;
        mdctx = EVP_MD_CTX_new();
        if (mdctx != NULL) {
            EVP_DigestInit_ex(mdctx, md, NULL);
            auto fileLen = lseek(modelFd, 0, SEEK_END);
            lseek(modelFd, 0, SEEK_SET);
            size_t bytesRead = 0, bytesToRead = fileLen;
            char buf[10240];
            while (bytesToRead > 0) {
                auto numBytes = read(modelFd, buf, 10240);
                if (numBytes > 0) {
                    EVP_DigestUpdate(mdctx, buf, numBytes);
                }
                bytesRead += numBytes;
                bytesToRead = fileLen - bytesRead;
            }
            EVP_DigestFinal_ex(mdctx, hashValue, &hashLen);
            EVP_MD_CTX_free(mdctx);
            lseek(modelFd, 0, SEEK_SET);
        } else {
            nnAssert("Unable to initialize context for computing SHA512 hash");
        }
    } else {
        nnAssert("Unable to get SHA512 digest");
    }

    std::stringstream hashString;
    for(unsigned int i=0; i < hashLen; i++)
        hashString << std::hex << static_cast<int>(hashValue[i]);

    return hashString.str();
}
#endif

#ifdef PERF_COUNTERS
// return GNA module frequency in MHz
float getGnaFrequencyMHz() {
    uint32_t eax = 1;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
    uint32_t family = 0;
    uint32_t model = 0;
    const uint8_t sixth_family = 6;
    const uint8_t cannon_lake_model = 102;
    const uint8_t gemini_lake_model = 122;
    const uint8_t ice_lake_model = 126;
    const uint8_t tiger_lake_model = 140;

    native_cpuid(&eax, &ebx, &ecx, &edx);
    family = (eax >> 8) & 0xF;

    // model is the concatenation of two fields
    // | extended model | model |
    // copy extended model data
    model = (eax >> 16) & 0xF;
    // shift
    model <<= 4;
    // copy model data
    model += (eax >> 4) & 0xF;

    if (family == sixth_family) {
        switch (model) {
            case cannon_lake_model:
            case ice_lake_model:
            case tiger_lake_model:
                return 400;
            case gemini_lake_model:
                return 200;
            default:
                return 1;
        }
    } else {
        // counters not supported and we returns just default value
        return 1;
    }
}
#endif
}
}
}
}
