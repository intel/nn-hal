#ifndef ANDROID_ML_NN_GNA_CPUOPS_H
#define ANDROID_ML_NN_GNA_CPUOPS_H

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
    float* output = nullptr;
    uint8_t* inputDataPtr;
    int32_t inputLen;
    std::string layerName;

    public:

        bool isCpuOp() {
            return true;
        }

        bool setInputData(void* dataPtr, uint32_t len) {
            inputDataPtr = static_cast<uint8_t*>(dataPtr);
            inputLen = len;

            VLOG(L1, "Setting input data for Dequantize OP");
            return true;
        }

        std::tuple<void*, int32_t> getOutputData() {
            VLOG(L1, "%s output len:%d", __func__, inputLen);
            return std::make_tuple((void*)output, inputLen);
        }

        DequantizeOp(std::string name, float sc, int32_t zp) : scale(sc), zeroPoint(zp) {
            inputDataPtr = nullptr;
            inputLen = 0;
            layerName = name;

            VLOG(L1, "%s scalefactor:%f zeropoint:%d", __func__, sc, zp);
        }

        bool  run() {
            const T* inputBuf = reinterpret_cast<const T*>(inputDataPtr);

            VLOG(L1, "%s inputLen:%d zeroPoint=%d scale:%f", __func__, inputLen, zeroPoint, scale);
            output = new float[inputLen];

            for(auto i =0; i < 4; i++)
                VLOG(L1, "Input at index:%d is :%d", i, inputBuf[i]);

            for (int i = 0; i < inputLen; ++i) {
                int32_t value = inputBuf[i];
                output[i] = static_cast<float>(scale * (value - zeroPoint));

                if (i < 4) {
                    VLOG(L1, "Index:%d input:%d Output:%f", i, value, output[i]);
                }
            }    

            return true;
        }

        float*  run(const uint8_t* inputData, const uint32_t len) {
            VLOG(L1, "%s ", __func__);
            output = new float[len];

            int32_t value;
            const T* inputBuf = reinterpret_cast<const T*>(inputData);
            for (int i = 0; i < len; ++i) {
                value = *(inputBuf + i);
                output[i] = static_cast<float>(scale * (value - zeroPoint));
            }

            return output;
        }

        std::string getLayerName() { return layerName; }

        ~DequantizeOp() {
            if (output) {
                delete output;
            }
        }
};

}
}
}
}
#endif