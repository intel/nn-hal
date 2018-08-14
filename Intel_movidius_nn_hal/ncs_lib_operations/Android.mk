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
LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)

LOCAL_SRC_FILES := fp.cpp ncs_lib.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../libncs/ncsdk-1.12.00.01/api/include \
                    $(LOCAL_PATH)/../graph_compiler_NCS \
                    $(LOCAL_PATH)
LOCAL_SHARED_LIBRARIES := libncsdk liblog libutils
LOCAL_CPPFLAGS := -fexceptions -o3
LOCAL_MODULE := libncs_nn_operation


include $(BUILD_SHARED_LIBRARY)
