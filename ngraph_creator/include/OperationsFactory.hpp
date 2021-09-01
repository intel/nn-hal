#pragma once

#include <Add.hpp>
#include <Average_Pool_2D.hpp>
#include <Batch_To_Space.hpp>
#include <Concat.hpp>
#include <Conv_2d.hpp>
#include <Depth_To_Space.hpp>
#include <Depthwise_Conv_2d.hpp>
#include <Dequantize.hpp>
#include <Div.hpp>
#include <Embedding_Lookup.hpp>
#include <Floor.hpp>
#include <FullyConnected.hpp>
#include <L2Pooling2D.hpp>
#include <L2_Normalization.hpp>
#include <LSTM.hpp>
#include <Logistic.hpp>
#include <Max_Pool_2d.hpp>
#include <Mean.hpp>
#include <Mul.hpp>
#include <Pad.hpp>
#include <RNN.hpp>
#include <Relu.hpp>
#include <Relu1.hpp>
#include <Relu6.hpp>
#include <Reshape.hpp>
#include <ResizeBilinear.hpp>
#include <Softmax.hpp>
#include <Space_To_Batch.hpp>
#include <Space_To_Depth.hpp>
#include <Squeeze.hpp>
#include <Strided_Slice.hpp>
#include <Sub.hpp>
#include <Tanh.hpp>
#include <Transpose.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class OperationsFactory {
private:
    std::shared_ptr<NgraphNodes> mNgraphNodes;

public:
    OperationsFactory(IntelDeviceType deviceType, std::shared_ptr<NnapiModelInfo> modelInfo,
                      std::shared_ptr<NgraphNodes> nodes);
    ~OperationsFactory();
    std::shared_ptr<OperationsBase> getOperation(int operationIndex,
                                                 const OperationType& operationType);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
