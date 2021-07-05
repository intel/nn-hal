#pragma once

#include <Abs.hpp>
#include <Add.hpp>
#include <Argmax.hpp>
#include <Argmin.hpp>
#include <Average_Pool_2D.hpp>
#include <Batch_To_Space.hpp>
#include <Cast.hpp>
#include <Channel_Shuffle.hpp>
#include <Concat.hpp>
#include <Conv_2d.hpp>
#include <Depthwise_Conv_2d.hpp>
#include <Dequantize.hpp>
#include <Div.hpp>
#include <Equal.hpp>
#include <Exp.hpp>
#include <Expand_Dims.hpp>
#include <Floor.hpp>
#include <FullyConnected.hpp>
#include <Gather.hpp>
#include <Greater.hpp>
#include <Greater_Equal.hpp>
#include <L2_Normalization.hpp>
#include <LSTM.hpp>
#include <Less.hpp>
#include <Less_Equal.hpp>
#include <Log.hpp>
#include <Log_Softmax.hpp>
#include <Logical_And.hpp>
#include <Logical_Not.hpp>
#include <Logical_Or.hpp>
#include <Logistic.hpp>
#include <Max_Pool_2d.hpp>
#include <Maximum.hpp>
#include <Mean.hpp>
#include <Minimum.hpp>
#include <Mul.hpp>
#include <Neg.hpp>
#include <Not_Equal.hpp>
#include <Pad.hpp>
#include <Pad_V2.hpp>
#include <Pow.hpp>
#include <Quantize.hpp>
#include <RNN.hpp>
#include <ROI_Pooling.hpp>
#include <RSQRT.hpp>
#include <Reduce_All.hpp>
#include <Reduce_Any.hpp>
#include <Reduce_Max.hpp>
#include <Reduce_Min.hpp>
#include <Reduce_Prod.hpp>
#include <Reduce_Sum.hpp>
#include <Relu.hpp>
#include <Relu1.hpp>
#include <Relu6.hpp>
#include <Reshape.hpp>
#include <SQRT.hpp>
#include <Select.hpp>
#include <Sin.hpp>
#include <Softmax.hpp>
#include <Space_To_Batch.hpp>
#include <Split.hpp>
#include <Squeeze.hpp>
#include <Sub.hpp>
#include <Tanh.hpp>
#include <Topk_V2.hpp>
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
