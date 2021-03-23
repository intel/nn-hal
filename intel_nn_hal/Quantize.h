#ifndef ANDROID_ML_NN_QUANTIZE_H
#define ANDROID_ML_NN_QUANTIZE_H

#include "Utils.h"
#include <type_traits>
#include <xmmintrin.h>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// TODO: Assumption is this is 1D vector (ignoring the batch size)
template<typename InputType, typename OutputType>
class QuantizeOp : public BaseOp {
    float scale;
    int32_t zeroPoint;
    void* output = nullptr;
    InputType* inputDataPtr;
    int32_t inputLen;
    std::string layerName;

    public:
        bool isCpuOp() {
            return true;
        }

        bool setInputData(void* dataPtr, uint32_t len) {
            inputDataPtr = static_cast<InputType*>(dataPtr);
            inputLen = len;

            VLOG(L1, "Setting input data for Quantize OP len:%d", len);
            return true;
        }

        std::tuple<void*, int32_t> getOutputData() {
            VLOG(L1, "%s output len:%d", __func__, inputLen);
            return std::make_tuple((void*)output, inputLen);
        }

        QuantizeOp(std::string name, float sc, int32_t zp) : scale(sc), zeroPoint(zp) {
            inputDataPtr = nullptr;
            inputLen = 0;
            layerName = name;
            VLOG(L1, "%s scalefactor:%f zeropoint:%d", __func__, sc, zp);
        }

        bool quantizeToQuant8Unsigned(const InputType* inputData, uint8_t* outputData) {
            //uint32_t len = getNumberOfElements(outShape.dimensions);

            for (uint32_t i = 0; i < inputLen; ++i) {
                outputData[i] = static_cast<uint8_t>(std::max<float>(0.0f, 
                                                    std::min<float>(255.0f, zeroPoint + std::round(inputData[i] /
                                                    scale))));

                if (i < 4)
                    VLOG(L1, "Index:%d inputdata:%f output:%d",i, inputData[i], outputData[i]);
            }
        }

        bool quantizeToQuant8Signed(const InputType* inputData, int8_t* outputData) {
            for (uint32_t i = 0; i < inputLen; ++i) {
                outputData[i] = static_cast<int8_t>(std::max<float>(-128.0f,
                                                    std::min<float>(127.0f, zeroPoint +
                                                    std::round(inputData[i] / scale))));
                
                if (i < 4)
                    VLOG(L1, "Index:%d inputdata:%f output:%d",i, inputData[i], outputData[i]);
            }
            return true;
        }

        bool  run() {
            VLOG(L1, "%s inputLen:%d zeroPoint=%d scale:%f", __func__, inputLen, zeroPoint, scale);
    
            for(auto i =0; i < 4; i++)
                VLOG(L1, "Input at index:%d is :%f", i, inputDataPtr[i]);
    
            if (std::is_same<OutputType, uint8_t>::value) {
                if (!output)
                    output = new uint8_t[inputLen];
    
                VLOG(L1, "Output type is of type uint8_t");
                quantizeToQuant8Unsigned(inputDataPtr, static_cast<uint8_t*>(output));
            } else if (std::is_same<OutputType, int8_t>::value) {
                if (!output)
                    output = new int8_t[inputLen];
    
                output = new int8_t[inputLen];
                VLOG(L1, "Output type is of type int8_t");
                quantizeToQuant8Signed(inputDataPtr, static_cast<int8_t*>(output));

                // auto iptr = static_cast<int8_t*>(output);
                // for(auto i =0; i < 4; i++)
                //     VLOG(L1, "Input at index:%d is :%f", i, iptr[i]);
            }

            VLOG(L1, "%s done >>>>>>>>>", __func__);
            return true;
        }

        std::string getLayerName() { return layerName; }

        ~QuantizeOp() {
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