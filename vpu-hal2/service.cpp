/*
 * Copyright (C) 2017 The Android Open Source Project
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

#include <hidl/HidlTransportSupport.h>
#include <hidl/LegacySupport.h>

#include "VpuDriver.h"


using android::hardware::configureRpcThreadpool;
using android::hardware::joinRpcThreadpool;
using android::hardware::neuralnetworks::V1_0::vpu_driver::VpuDriver;

 int main(int /* argc */, char* /* argv */ []) {
//     android::sp<VpuDriver> vpu = new VpuDriver("vpu-driver");
     android::sp<VpuDriver> vpu = new VpuDriver();

     configureRpcThreadpool(4, true);
     android::status_t status = vpu->registerAsService("myriad-vpu");
     LOG_ALWAYS_FATAL_IF(
             status != android::OK,
             "Error while registering as service: %d", status);
     joinRpcThreadpool();

     return 0;
 }
