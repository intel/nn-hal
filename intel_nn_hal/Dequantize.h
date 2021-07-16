#ifndef ANDROID_ML_NN_DEQUANTIZE_H
#define ANDROID_ML_NN_DEQUANTIZE_H

#include "Utils.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// TODO: Assumption is this is 1D vector (ignoring the batch size)
template<typename T>
class DequantizeOp : public BaseOp {
    float scale;
    int32_t zeroPoint;
    void* output = nullptr;
    float* deq_output = nullptr;
    void* inputDataPtr;
    int32_t inputLen;
    std::string layerName;
    std::map<uint32_t, uint32_t> mGraphtoOpIndex;
    bool dummyOp_ = false;
    bool isSubgraphInput = false;
    public:

        bool isCpuOp() {
            return true;
        }
        void setSubgraphInput() {
            isSubgraphInput = true;
        }
         bool hasSubgraphInput() {
            return isSubgraphInput;
        }

        bool setInputIndex(uint32_t graph_index, uint32_t op_index) {
            mGraphtoOpIndex[graph_index] = op_index;
            return true;
        }

        bool setInputData(uint32_t graph_index, void* dataPtr, uint32_t len) {
            if (mGraphtoOpIndex[graph_index] == 0) {
                inputDataPtr = dataPtr;
                inputLen = len;
                return true;
            }
            else {
                nnAssert("Cannot have Operand index greater or equal to one\n");
            }
            return false;
        }

        std::tuple<void*, int32_t> getOutputData() {
            return std::make_tuple((void*)output, inputLen);
        }

        DequantizeOp(std::string name, float sc, int32_t zp, bool dummyOp) : scale(sc), zeroPoint(zp) {
            inputDataPtr = nullptr;
            inputLen = 0;
            layerName = name;
            dummyOp_ = dummyOp;
        }

        bool  run() {
            if (dummyOp_) {
                output = (void*)inputDataPtr;
                return true;
            }
            const T* inputBuf = reinterpret_cast<const T*>(inputDataPtr);
            deq_output = new float[inputLen];

            for (int i = 0; i < inputLen; ++i) {
                int32_t value = inputBuf[i];
                deq_output[i] = static_cast<float>(scale * (value - zeroPoint));
            }
            output = deq_output;

            return true;
        }

        std::string getLayerName() { return layerName; }

        void cleanup() {
            if (deq_output && !dummyOp_)
                delete[] deq_output;
        }

        ~DequantizeOp() {
            if (deq_output && !dummyOp_)
                delete[] deq_output;
        }
};

}
}
}
}
#endif
