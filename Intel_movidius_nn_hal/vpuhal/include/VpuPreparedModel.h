/*
 * Copyright (C) 2018 The Android Open Source Project
 * Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef ANDROID_ML_NN_VPU_PREPAREDMODEL_H
#define ANDROID_ML_NN_VPU_PREPAREDMODEL_H


/*
#include <android/hardware/neuralnetworks/1.0/IPreparedModel.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hidlmemory/mapping.h>
#include <hardware/hardware.h>
*/

#include <sys/mman.h>
#include <string>
#include <iostream>

#include "HalInterfaces.h"
#include "NeuralNetworks.h"
#include "VpuExecutor.h" //TODO create this file


using ::android::hidl::memory::V1_0::IMemory;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {


class VpuPreparedModel : public IPreparedModel {

  public:
      static int network_count_ex;
      VpuPreparedModel(const Model& model)
            : // Make a copy of the model, as we need to preserve it.
              mModel(model) {network_count_ex++;}
      ~VpuPreparedModel() override {deinitialize();}
      bool initialize(const Model& model);
      Return<ErrorStatus> execute(const Request& request,
                                  const sp<IExecutionCallback>& callback) override;
      static bool isOperationSupported(const Operation& operation, const Model& model);
      static bool validModel(const Model& model);  //TODO Utils.cpp validateModel was changed to validModel


private:
        void deinitialize();
        Operation_inputs_info get_operation_operands_info_model(const Model& model, const Operation& operation);
        void asyncExecute(const Request& request, const sp<IExecutionCallback>& callback);

        Model mModel;
        std::vector<RunTimePoolInfo> mPoolInfos;
};



}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android


#endif //ANDROID_ML_NN_VPU_PREPAREDMODEL_H
