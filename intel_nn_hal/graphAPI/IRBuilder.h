#ifndef ANDROID_ML_NN_BUILDER_H
#define ANDROID_ML_NN_BUILDER_H

#pragma once
#include <android/hardware/neuralnetworks/1.2/types.h>
#include "IRLayers.h"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
namespace IRBuilder {

namespace BuilderFCLayer 
{
    struct ParamsData {
        IRBlob::Ptr data;
        bool lifeTime;
    };
    struct FCParams {
        ParamsData weights;
        ParamsData bias;
    };
} // Builder FC layer

namespace LstmLayer 
{
    enum class LstmImpl : char {
        PROJECTION = 1,
        PEEPHOLE = 2,
        CIFG = 4
    };

    enum class LstmActivationFn: char {
        NONE = 0,
        RELU = 1,
        RELU6 = 3,
        TANH = 4,
        SIGMOID = 6
    };

    struct LstmCellDescription {
        int clippingThresholdCellState;
        int clippingThresholdProjState;
        int lstmImplementation;
    };

    struct LstmCellData {
        IRBlob::Ptr data;
        bool lifeTime;
    };

    // TODO: Make these const operands
    struct LstmParams {
        LstmCellData input2inputWeights;
        LstmCellData input2ForgetWeights;
        LstmCellData input2CellWeights;
        LstmCellData input2OutputWeights;
        LstmCellData recurrant2inputWeights;
        LstmCellData recurrant2ForgetWeights;
        LstmCellData recurrant2CellWeights;
        LstmCellData recurrant2OutputWeights;
        LstmCellData cell2InputWeights;
        LstmCellData cell2ForgetWeights;
        LstmCellData cell2OutputWeights;
        LstmCellData inputGateBias;
        LstmCellData outputGateBias;
        LstmCellData forgetGateBias;
        LstmCellData cellBias;
        LstmCellData projectionWeights;
        LstmCellData projectionBias;
        LstmCellData inputLayerNormWeights;
        LstmCellData outputLayerNormWeights;
        LstmCellData forgetLayerNormWeights;
        LstmCellData cellLayerNormWeights;

        bool isLayerNormSupported;
        int  activationFunction;
    };

    // TODO: fix the return
} // LstmLayer namespace

class BuilderNetwork;

class ModelBuilder {
    public:
        BuilderNetwork* getBuilderNetwork() {
            return mBuilder;
        };

        void addOutputLayer();
        void setLayerData(IRBlob::Ptr dataToSet, int idx, IRBlob::Ptr destToSet) ;
        int check4LayerData(IRBlob::Ptr blob);
        void initializeBuilder();

        OutputPort createFC(BuilderFCLayer::FCParams& params, IRBlob::Ptr input);
        IRBlob::Ptr generateBlobwithData(InferenceEngine::SizeVector dims, 
                                            InferenceEngine::Layout layout, 
                                            std::vector<std::vector<float>> data_to_set);
        OutputPort createFullLstm(LstmLayer::LstmParams& params, LstmLayer::LstmCellDescription& lstmDesc, 
                                    IRBlob::Ptr input, IRBlob::Ptr cellStateIn, IRBlob::Ptr outputStateIn);
        std::shared_ptr<InferenceEngine::ICNNNetwork> convertBuilder();
        
        void addToBlobLayerMap(IRBlob::Ptr blob, int index) {
            mBlob2LayerIdxMap[blob] = index;
        }
        int layer_name_count = 0;

        std::map<IRBlob::Ptr, int> mBlob2LayerIdxMap;
        IRBlob::Ptr mInGateBlob;
        
    private:
        BuilderNetwork* mBuilder;
        
};
}
}
}
}
}

#endif