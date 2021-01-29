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

#ifndef ANDROID_ML_NN_GNA_PREPAREDMODEL_H
#define ANDROID_ML_NN_GNA_PREPAREDMODEL_H

#include <android/hardware/neuralnetworks/1.2/IPreparedModel.h>
#include <android/hardware/neuralnetworks/1.2/types.h>
#include <android/hidl/memory/1.0/IMemory.h>
#include <hardware/hardware.h>
#include <hidlmemory/mapping.h>
#include <sys/mman.h>
#include <fstream>
#include <string>

#include "BasePreparedModel.h"

using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class GnaPreparedModel : public BasePreparedModel {
public:
    GnaPreparedModel(const Model& model) : BasePreparedModel("GNA", model) {}
    ~GnaPreparedModel() { deinitialize(); }

    bool initialize(const Model& model) override;
    Blob::Ptr GetConstWeightsOperandAsTensor(uint32_t index, const Model& model) override;
    Blob::Ptr GetConstOperandAsTensor(int operand_index, int operation_idx,
                                      const Model& model) override;
    Blob::Ptr GetInOutOperandAsBlob(RunTimeOperandInfo& op, const uint8_t* buf,
                                    uint32_t& len) override;

protected:
    void deinitialize() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // ANDROID_ML_NN_GNA_PREPAREDMODEL_H
