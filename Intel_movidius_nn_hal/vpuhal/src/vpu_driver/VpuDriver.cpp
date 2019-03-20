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

#define LOG_TAG "VpuDriver"

#include <android-base/logging.h>
#include <android-base/properties.h>
#include <android-base/strings.h>
#include <sys/system_properties.h>
#include <hidl/LegacySupport.h>
#include <thread>

#include "VpuDriver.h"
#include "VpuUtils.h"
#include "VpuPreparedModel.h"
#include "HalInterfaces.h"

#include "ncs_lib.h"
#define NCS_NUM 1

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace V1_0 {
namespace vpu_driver {


//getCapabilities() function

Return<void> VpuDriver::getCapabilities(getCapabilities_cb cb) {
      Capabilities capabilities = {.float32Performance = {.execTime = 0.5f, .powerUsage = 0.5f},
                                   .quantized8Performance = {.execTime = 1.0f, .powerUsage = 0.7f}};
      cb(ErrorStatus::NONE, capabilities);
      return Void();
}

//getSupportedOperations() function

Return<void> VpuDriver::getSupportedOperations(const Model& model,
                                                    getSupportedOperations_cb cb) {
      int count = model.operations.size();
      std::vector<bool> supported(count, false);

      if (!VpuPreparedModel::validModel(model)) {
          ALOGE("model is not valid");
          std::vector<bool> supported;
          cb(ErrorStatus::INVALID_ARGUMENT, supported);
          return Void();
      }
      else
      {
        const char kVLogPropKey[] = "nn.vpu.disable";

        const uint32_t vLogSetting = getProp(kVLogPropKey);
        ALOGD("vLogSetting: %d", vLogSetting);

        if(vLogSetting == 1){
          for (int i = 0; i < count; i++) {
              const auto& operation = model.operations[i];
              supported[i] = false;
          }
        }
        else{
          for (int i = 0; i < count; i++) {
              const auto& operation = model.operations[i];
              supported[i] = VpuPreparedModel::isOperationSupported(operation, model);
            }
        }
        cb(ErrorStatus::NONE, supported);
      }
      return Void();
}

//prepareModel() function

Return<ErrorStatus> VpuDriver::prepareModel(const Model& model,
                                               const sp<IPreparedModelCallback>& callback) {

    int devStatus;
    devStatus = ncs_init();
    if (devStatus){
        ALOGE("Error - VpuDriver returning since NN device is disconnected/offline.");
        return ErrorStatus::DEVICE_UNAVAILABLE;
    }

    if (VLOG_IS_ON(DRIVER)) {
      VLOG(DRIVER) << "VpuDriver::prepareModel begin";
      logModelToInfo(model);
    }

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to prepareModel");
        return ErrorStatus::INVALID_ARGUMENT;
    }

    if (!VpuPreparedModel::validModel(model)) {
        ALOGE("model is not valid");
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<VpuPreparedModel> preparedModel = new VpuPreparedModel(model);
    if (!preparedModel->initialize(model)) {
        ALOGE("failed to initialize preparedmodel");
        callback->notify(ErrorStatus::GENERAL_FAILURE, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    callback->notify(ErrorStatus::NONE, preparedModel);
    return ErrorStatus::NONE;
}

//getStatus() function

Return<DeviceStatus> VpuDriver::getStatus() {

  int status;
  ALOGD("VpuDriver getStatus()");
  status = ncs_init();
  if (!status){
    return DeviceStatus::OFFLINE;
    ALOGE("VPU Device Unavilable");
  }else{
    ALOGD("VPU Device avilable");
    return DeviceStatus::AVAILABLE;
  }
}
/*
int VpuDriver::run() {
    android::hardware::configureRpcThreadpool(4, true);
    if (registerAsService("vpudriver") != android::OK) { //TODO check if the service name is vpudriver only?
        LOG(ERROR) << "Could not register VPU service";
        return 1;
    }
    android::hardware::joinRpcThreadpool();
    LOG(ERROR) << "Service exited!";
    return 1;
}
*/

}  // namespace vpu_driver
}  // namespace V1_0
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
