#pragma once

#include <Abs.hpp>
#include <Add.hpp>
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
#include <LSTM.hpp>
#include <Reshape.hpp>
#include <Softmax.hpp>

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
