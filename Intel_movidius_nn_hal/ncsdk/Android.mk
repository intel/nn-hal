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

LOCAL_MODULE := libmvnc

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
                    libmvnc

include $(BUILD_EXECUTABLE)
