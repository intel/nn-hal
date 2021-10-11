#pragma once

#include <Abs.hpp>
#include <Add.hpp>
#include <Argmax.hpp>
#include <Argmin.hpp>
#include <AveragePool2D.hpp>
#include <BatchToSpace.hpp>
#include <BidirectionalSequenceRNN.hpp>
#include <Cast.hpp>
#include <ChannelShuffle.hpp>
#include <Concat.hpp>
#include <Conv2d.hpp>
#include <DepthToSpace.hpp>
#include <DepthwiseConv2d.hpp>
#include <Dequantize.hpp>
#include <Div.hpp>
#include <EmbeddingLookup.hpp>
#include <Equal.hpp>
#include <Exp.hpp>
#include <ExpandDims.hpp>
#include <Floor.hpp>
#include <FullyConnected.hpp>
#include <Gather.hpp>
#include <Greater.hpp>
#include <GreaterEqual.hpp>
#include <GroupedConv2d.hpp>
#include <HardSwish.hpp>
#include <InstanceNormalization.hpp>
#include <L2Normalization.hpp>
#include <L2Pooling2D.hpp>
#include <LSTM.hpp>
#include <Less.hpp>
#include <LessEqual.hpp>
#include <Log.hpp>
#include <LogSoftmax.hpp>
#include <LogicalAnd.hpp>
#include <LogicalNot.hpp>
#include <LogicalOr.hpp>
#include <Logistic.hpp>
#include <MaxPool2d.hpp>
#include <Maximum.hpp>
#include <Mean.hpp>
#include <Minimum.hpp>
#include <Mul.hpp>
#include <Neg.hpp>
#include <NotEqual.hpp>
#include <PRelu.hpp>
#include <Pad.hpp>
#include <PadV2.hpp>
#include <Pow.hpp>
#include <Quantize.hpp>
#include <RNN.hpp>
#include <ROIAlign.hpp>
#include <ROIPooling.hpp>
#include <RSQRT.hpp>
#include <ReduceAll.hpp>
#include <ReduceAny.hpp>
#include <ReduceMax.hpp>
#include <ReduceMin.hpp>
#include <ReduceProd.hpp>
#include <ReduceSum.hpp>
#include <Relu.hpp>
#include <Relu1.hpp>
#include <Relu6.hpp>
#include <Reshape.hpp>
#include <ResizeBilinear.hpp>
#include <ResizeNearestNeighbor.hpp>
#include <SQRT.hpp>
#include <Select.hpp>
#include <Sin.hpp>
#include <Softmax.hpp>
#include <SpaceToBatch.hpp>
#include <SpaceToDepth.hpp>
#include <Split.hpp>
#include <Squeeze.hpp>
#include <StridedSlice.hpp>
#include <Sub.hpp>
#include <Tanh.hpp>
#include <TopkV2.hpp>
#include <Transpose.hpp>
#include <TransposeConv2D.hpp>
#include <UnidirectionalSequenceRNN.hpp>

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
