#ifndef ANDROID_ML_NN__HAL_INTERFACES_H
#define ANDROID_ML_NN__HAL_INTERFACES_H

#include <android/hardware/neuralnetworks/1.0/IDevice.h>
#include <android/hardware/neuralnetworks/1.0/IExecutionCallback.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.0/IPreparedModelCallback.h>
#include <android/hardware/neuralnetworks/1.0/types.h>
//#include <android/hidl/allocator/1.0/IAllocator.h>
//#include <android/hidl/memory/1.0/IMemory.h>
#include "IAllocator.h"
#include "IMemory.h"
//#include <hidlmemory/mapping.h>
#include "mapping.h"

using ::android::hardware::Return;
using ::android::hardware::Void;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
using ::android::hardware::hidl_vec;
using ::android::hardware::neuralnetworks::V1_0::Capabilities;
using ::android::hardware::neuralnetworks::V1_0::DataLocation;
using ::android::hardware::neuralnetworks::V1_0::DeviceStatus;
using ::android::hardware::neuralnetworks::V1_0::ErrorStatus;
using ::android::hardware::neuralnetworks::V1_0::FusedActivationFunc;
using ::android::hardware::neuralnetworks::V1_0::IDevice;
using ::android::hardware::neuralnetworks::V1_0::IExecutionCallback;
using ::android::hardware::neuralnetworks::V1_0::IPreparedModel;
using ::android::hardware::neuralnetworks::V1_0::IPreparedModelCallback;
using ::android::hardware::neuralnetworks::V1_0::Model;
using ::android::hardware::neuralnetworks::V1_0::Operand;
using ::android::hardware::neuralnetworks::V1_0::OperandLifeTime;
using ::android::hardware::neuralnetworks::V1_0::OperandType;
using ::android::hardware::neuralnetworks::V1_0::Operation;
using ::android::hardware::neuralnetworks::V1_0::OperationType;
using ::android::hardware::neuralnetworks::V1_0::PerformanceInfo;
using ::android::hardware::neuralnetworks::V1_0::Request;
using ::android::hardware::neuralnetworks::V1_0::RequestArgument;
using ::android::hidl::allocator::V1_0::IAllocator;
using ::android::hidl::memory::V1_0::IMemory;


namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {

}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
#endif // ANDROID_ML_NN__HAL_INTERFACES_H
