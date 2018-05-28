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
#define LOG_TAG "neuralnetworks-vpu"

#include "VpuUtils.h"

#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <hidl/HidlTransportSupport.h>
#include <thread>

#include "VpuDriver.h"

using android::hardware::configureRpcThreadpool;
using android::hardware::joinRpcThreadpool;
using android::hardware::neuralnetworks::V1_0::vpu_driver::VpuDriver;


int main() {
     android::sp<VpuDriver> vpu = new VpuDriver();
     configureRpcThreadpool(4, true);
     android::status_t status = vpu->registerAsService("vpudriver");
     if (status == android::OK) {
             ALOGI("VPU HAL Ready.");
             joinRpcThreadpool();
         }
    ALOGE("Cannot register VPU HAL service");
    return 0;
}
