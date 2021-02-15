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

#include "Driver.h"
#include <iostream>

namespace android {
namespace hardware {
namespace neuralnetworks {

// static
// This registers the SampleDriverFull into the DeviceManager.
::android::sp<V1_0::IDevice> V1_0::IDevice::getService(const std::string& /*serviceName*/,
                                           bool /*dummy*/) {
  return new nnhal::Driver("GNA");
}

}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
