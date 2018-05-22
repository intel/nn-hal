LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)

#PATH_TO_LIBUSB_SRC:= $(LOCAL_PATH)/../../../../../../../../../external/libusb


# libmvnc
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)


LIBUSB_HEADER:= $(LOCAL_PATH)/../../../../../../../../../external/libusb/libusb

#LIBUSB_HEADER:= $(LIBUSB_ROOT_ABS)

MV_COMMON_BASE:= $(LOCAL_PATH)/common
XLINK_BASE:= $(MV_COMMON_BASE)/components/XLink
XLINKCONSOLE_BASE:= $(MV_COMMON_BASE)/components/XLinkConsole
XLINK_CFLAGS:= -I$(XLINK_BASE)/shared \
								-I$(XLINK_BASE)/pc \
								-I$(XLINKCONSOLE_BASE)/pc \
								-I$(MV_COMMON_BASE)/swCommon/include \
								-I$(MV_COMMON_BASE)/shared/include


LOCAL_MODULE := libmvnc
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := 64
LOCAL_MULTILIB := both
LOCAL_MODULE_OWNER := intel
LOCAL_SRC_FILES := \
	mvnc_api.c \
	fp16.c \
	mvnc_api_highclass.c \
	common/components/XLink/pc/UsbLinkPlatform.cpp \
	common/components/XLink/pc/usb_boot.c \
	common/components/XLink/shared/XLink.c \
	common/components/XLink/shared/XLinkDispatcher.c \
	common/components/XLinkConsole/pc/XLinkConsole.c


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
	$(LOCAL_PATH)/../include \
	$(LIBUSB_HEADER)

#LOCAL_C_INCLUDES += $(LIBUSB_ROOT_ABS)

LOCAL_CFLAGS += $(XLINK_CFLAGS) -D__PC__ -DUSE_USB_VSC -DDEVICE_SHELL_ENABLED -Wno-error
LOCAL_CFLAGS += -O2 -Wall -pthread -fPIC -MMD -MP

LOCAL_SHARED_LIBRARIES := libusb1.0 liblog

include $(BUILD_SHARED_LIBRARY)

#include $(BUILD_STATIC_LIBRARY)
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2450.mvcmd
LOCAL_SRC_FILES := mvnc/MvNCAPI-ma2450.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)

#include $(PATH_TO_LIBUSB_SRC)/android/jni/libusb.mk
