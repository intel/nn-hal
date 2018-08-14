#
# Copyright (C) 2018 The Android Open Source Project
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := android.hardware.neuralnetworks@1.0-vpudriver-impl
LOCAL_PROPRIETARY_MODULE := true
LOCAL_SRC_FILES := \
    src/vpu_driver/VpuDriver.cpp \
    src/vpu_driver/VpuPreparedModel.cpp \
		src/vpu_driver/VpuExecutor.cpp \
    src/vpu_driver/VpuUtils.cpp \
		src/vpu_operations/VpuOperationsUtils.cpp \
		src/vpu_operations/VpuActivation.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
  $(LOCAL_PATH)/include \
  $(LOCAL_PATH)/../libncs/ncsdk-1.12.00.01/api/include \
  $(LOCAL_PATH)/../ncs_lib_operations \
	$(LOCAL_PATH)/../graph_compiler_NCS \
	frameworks/ml/nn/runtime/include


#TODO update the CFLAGS

LOCAL_CFLAGS += -fexceptions

LOCAL_SHARED_LIBRARIES := \
                    libhidlbase \
                    libhidltransport \
                    libutils \
                    liblog \
                    libcutils \
                    libhardware \
                    libbase \
                    libcutils \
                    libhidlmemory \
                    android.hardware.neuralnetworks@1.0 \
                    android.hidl.allocator@1.0 \
                    libncsdk \
                    libncs_nn_operation \
										libncs_graph_compiler \
                    android.hidl.memory@1.0

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := android.hardware.neuralnetworks@1.0-service-vpudriver
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.0-vpudriver.rc
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
LOCAL_SRC_FILES := service.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
  $(LOCAL_PATH)/include \
  $(LOCAL_PATH)/../ncsdk/include \
  $(LOCAL_PATH)/../ncs_lib_operations \
	frameworks/ml/nn/runtime/include

LOCAL_CFLAGS += -fexceptions

LOCAL_SHARED_LIBRARIES := \
            libhidlbase \
            libhidltransport \
            libutils \
            liblog \
            libcutils \
            libhardware \
            libhidlmemory \
            android.hardware.neuralnetworks@1.0 \
            android.hidl.allocator@1.0 \
            android.hidl.memory@1.0 \
            android.hardware.neuralnetworks@1.0-vpudriver-impl

include $(BUILD_EXECUTABLE)
