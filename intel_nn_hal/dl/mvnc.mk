LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LIBUSB_HEADER:= $(LOCAL_PATH)/../../../../../../../external/libusb/libusb

MV_COMMON_BASE:= $(LOCAL_PATH)/inference-engine/thirdparty/movidius
XLINK_BASE:= $(MV_COMMON_BASE)/XLink
#XLINKCONSOLE_BASE:= $(MV_COMMON_BASE)/components/XLinkConsole

LOCAL_MODULE := libmvnc
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := 64
LOCAL_MULTILIB := both
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/movidius/mvnc/src/mvnc_api.c \
	inference-engine/thirdparty/movidius/mvnc/src/fp16.c \
	inference-engine/thirdparty/movidius/XLink/pc/UsbLinkPlatform.c \
	inference-engine/thirdparty/movidius/XLink/pc/usb_boot.c \
	inference-engine/thirdparty/movidius/XLink/shared/XLink.c \
	inference-engine/thirdparty/movidius/XLink/shared/XLinkDispatcher.c \

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/shared/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/pc \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/shared \
	$(LIBUSB_HEADER) \
	$(LOCAL_PATH)/../../../../../../../external/libusb \
	$(LOCAL_PATH)/../../../../../../../external/libusb/libusb

XLINK_CFLAGS:= -I$(XLINK_BASE)/shared \
							 -I$(XLINK_BASE)/pc \
							 -I$(MV_COMMON_BASE)/shared/include

LOCAL_CFLAGS += \
	-fvisibility=default \
	-Wno-error \
	-O2 \
	-Wall \
	-pthread \
	-fPIE \
	-fPIC \
	-MMD \
	-MP \
	-Wformat \
	-Wformat-security \
	-fstack-protector-strong \
	-D_FORTIFY_SOURCE=2

LOCAL_CFLAGS += \
	-DENABLE_MYRIAD=1 \
	-DENABLE_OBJECT_DETECTION_TESTS=1 \
	-DENABLE_SEGMENTATION_TESTS=1 \
	-DHAVE_STRUCT_TIMESPEC \
	-DDEVICE_SHELL_ENABLED \
	-DUSE_USB_VSC \
	-D_CRT_SECURE_NO_WARNINGS \
	-D__PC__ \
	-D__ANDROID__

LOCAL_SHARED_LIBRARIES := libusb1.0 liblog

include $(BUILD_SHARED_LIBRARY)
####################################################
#include $(BUILD_STATIC_LIBRARY)
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2450.mvcmd
LOCAL_SRC_FILES := inference-engine/temp/vpu/firmware/ma2450/mvnc/MvNCAPI-ma2450.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
#####################################################
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2480.mvcmd
LOCAL_SRC_FILES := inference-engine/temp/vpu/firmware/ma2480/mvnc/MvNCAPI-ma2480.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
####################################################
#include $(PATH_TO_LIBUSB_SRC)/android/jni/libusb.mk
