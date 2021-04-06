#ifndef ANDROID_ML_NN_EMBEDDINGLOOKUP_H
#define ANDROID_ML_NN_EMBEDDINGLOOKUP_H

#include "Utils.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class EmbeddingLookupOp : public BaseOp {
    float* output = nullptr;
    float* valuesDataPtr;
    int32_t valuesLen;
    std::vector<uint32_t> values_dims_;
    std::vector<uint32_t> lookup_dims_;
    OperandType valueType_;
    int32_t* lookupDataPtr;
    uint32_t lookupLen;
    uint32_t outputLen;
    std::string layerName;
    std::map<uint32_t, uint32_t> mGraphtoOpIndex;
    std::map<int32_t, std::tuple<float*, int32_t>> indexToInputLenMap;
    std::map<int32_t, float*> indexToOutputMap;

    public:

        bool isCpuOp() {
            return true;
        }

        bool setInputIndex(uint32_t graph_index, uint32_t op_index) override {
            mGraphtoOpIndex[graph_index] = op_index;
            return true;
        }

        bool setInputData(uint32_t graph_index, void* dataPtr, uint32_t len) override {
            auto opIndex = mGraphtoOpIndex[graph_index];
            if(opIndex > 1) {
                nnAssert("Cannot have Operand index greater than one\n");
                return false;
            }
            if(opIndex == 0) {
                lookupDataPtr = static_cast<int32_t*>(dataPtr);
                lookupLen = len;
            }
            else if(opIndex == 1) {
                valuesDataPtr  = static_cast<float*>(dataPtr);
                valuesLen = len;
            }
            return true;
        }

        std::tuple<void*, int32_t> getOutputData() {
            return std::make_tuple((void*)output, outputLen);
        }

        EmbeddingLookupOp(std::string name, std::vector<uint32_t> values_dims, std::vector<uint32_t> lookup_dims, OperandType valueType) {
            layerName = name;
            values_dims_ = values_dims;
            lookup_dims_ = lookup_dims;
            valueType_ = valueType;
        }

        bool  run() {
            uint32_t row_size = values_dims_[0];
            uint32_t valueBytes = sizeOfData(valueType_, values_dims_);

            auto row_bytes = valueBytes / row_size;
            uint32_t idx;
            for( uint32_t i = 0; i < lookup_dims_[0]; i++) {
                idx = *(lookupDataPtr);
            }

            output = new float[row_size];
            memcpy(output, (uint8_t*)valuesDataPtr + idx * row_bytes, row_bytes);
            outputLen = row_bytes / sizeof(float);
            return true;
        }

        std::string getLayerName() { return layerName; }

		void cleanup() {
			if (output)
				delete output;
		}

        ~EmbeddingLookupOp() {
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