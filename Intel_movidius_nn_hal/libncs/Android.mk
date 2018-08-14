LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)

LIBUSB_HEADER:=

LOCAL_SRC_FILES := \
	ncsdk-1.12.00.01/api/src/usb_boot.c \
	ncsdk-1.12.00.01/api/src/usb_link_vsc.c \
	ncsdk-1.12.00.01/api/src/mvnc_api.c

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/ncsdk-1.12.00.01/api/src \
	$(LOCAL_PATH)/ncsdk-1.12.00.01/api/include \
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

LOCAL_SRC_FILES := ncsdk-1.12.00.01/api/src/mvnc/MvNCAPI.mvcmd

include $(BUILD_PREBUILT)



include $(CLEAR_VARS)
LOCAL_MODULE := ncs_test1_app
LOCAL_SRC_FILES := ncsdk-1.12.00.01/examples/apps/hello_ncs_cpp/cpp/hello_ncs.cpp

LOCAL_C_INCLUDES +=  \
                 $(LOCAL_PATH) \
                 $(LOCAL_PATH)/ncsdk-1.12.00.01/api/include

LOCAL_CFLAGS += -fexceptions

LOCAL_SHARED_LIBRARIES := \
                    libutils \
                    liblog \
                    libcutils \
                    libncsdk

include $(BUILD_EXECUTABLE)
