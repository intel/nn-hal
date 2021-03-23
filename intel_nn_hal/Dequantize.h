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
            return true;
        }

        std::tuple<void*, int32_t> getOutputData() {
            return std::make_tuple((void*)output, inputLen);
        }

        DequantizeOp(std::string name, float sc, int32_t zp) : scale(sc), zeroPoint(zp) {
            inputDataPtr = nullptr;
            inputLen = 0;
            layerName = name;
        }

        bool  run() {
            const T* inputBuf = reinterpret_cast<const T*>(inputDataPtr);
            output = new float[inputLen];

            for (int i = 0; i < inputLen; ++i) {
                int32_t value = inputBuf[i];
                output[i] = static_cast<float>(scale * (value - zeroPoint));
            }

            return true;
        }

        float*  run(const uint8_t* inputData, const uint32_t len) {
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

		void cleanup() {
			if (output)
				delete output;
		}

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
