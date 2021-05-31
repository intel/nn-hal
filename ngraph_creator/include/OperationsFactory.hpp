#pragma once

#include <Abs.hpp>
#include <Add.hpp>
#include <Average_Pool_2D.hpp>
#include <Cast.hpp>
#include <Concat.hpp>
#include <Conv_2d.hpp>
#include <Depthwise_Conv_2d.hpp>
#include <Div.hpp>
#include <Equal.hpp>
#include <Exp.hpp>
#include <Floor.hpp>
#include <Gather.hpp>
#include <Greater.hpp>
#include <Greater_Equal.hpp>
#include <LSTM.hpp>
#include <Less.hpp>
#include <Less_Equal.hpp>
#include <Log.hpp>
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
#include <Pow.hpp>
#include <Reduce_All.hpp>
#include <Reduce_Any.hpp>
#include <Reduce_Min.hpp>
#include <Reduce_Prod.hpp>
#include <Relu.hpp>
#include <Relu1.hpp>
#include <Relu6.hpp>
#include <Reshape.hpp>
#include <SQRT.hpp>
#include <Sin.hpp>
#include <Softmax.hpp>
#include <Split.hpp>
#include <Squeeze.hpp>
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
    OperationsFactory(const std::string& plugin, std::shared_ptr<NnapiModelInfo> modelInfo,
                      std::shared_ptr<NgraphNodes> nodes);
    ~OperationsFactory();
    std::shared_ptr<OperationsBase> getOperation(int operationIndex,
                                                 const OperationType& operationType);
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
