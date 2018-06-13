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

LIBUSB_HEADER:=

LOCAL_SRC_FILES := \
	src/usb_boot.c \
	src/usb_link_vsc.c \
	src/mvnc_api.c

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/src \
	$(LOCAL_PATH)/include \
	external/libusb/libusb
LOCAL_CFLAGS += -O2 -Wall -pthread -fPIC -MMD -MP

LOCAL_SHARED_LIBRARIES := libusb

LOCAL_MODULE := libncsdk

include $(BUILD_SHARED_LIBRARY)


$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI.mvcmd

LOCAL_MODULE_TAGS := optional

LOCAL_MODULE_CLASS := ETC
LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

LOCAL_SRC_FILES := src/mvnc/MvNCAPI.mvcmd

include $(BUILD_PREBUILT)


include $(CLEAR_VARS)
LOCAL_MODULE := ncs_test1_app
LOCAL_SRC_FILES := ncs_tests/ncs_test1.cpp

LOCAL_C_INCLUDES +=  \
                 $(LOCAL_PATH) \
                 $(LOCAL_PATH)/../ncsdk/include

LOCAL_CFLAGS += -fexceptions

LOCAL_SHARED_LIBRARIES := \
                    libutils \
                    liblog \
                    libcutils \
                    libncsdk

include $(BUILD_EXECUTABLE)
