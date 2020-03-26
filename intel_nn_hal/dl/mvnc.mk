LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LIBUSB_HEADER:= $(LOCAL_PATH)/../../../../../external/libusb/libusb

MV_COMMON_BASE:= $(LOCAL_PATH)/inference-engine/thirdparty/movidius
XLINK_BASE:= $(MV_COMMON_BASE)/XLink
#XLINKCONSOLE_BASE:= $(MV_COMMON_BASE)/components/XLinkConsole

LOCAL_MODULE := libmvnc
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/movidius/mvnc/src/mvnc_api.c \
	inference-engine/thirdparty/movidius/mvnc/src/mvnc_data.c \
	inference-engine/thirdparty/movidius/mvnc/src/watchdog/watchdog.cpp \
	# inference-engine/thirdparty/movidius/USB_WIN/gettime.c \
	# inference-engine/thirdparty/movidius/USB_WIN/usb_winusb.c \
	# inference-engine/thirdparty/movidius/WinPthread/win_pthread.c \
	# inference-engine/thirdparty/movidius/WinPthread/win_semaphore.c \

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include/watchdog \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/shared \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/shared/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/watchdog \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/pc \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/shared \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/shared/include \
	$(LOCAL_PATH)/inference-engine/src/vpu/common/include/vpu \
	$(LIBUSB_HEADER) \
	$(LOCAL_PATH)/../../../../../external/libusb \
	$(LOCAL_PATH)/../../../../../external/libusb/libusb \
	# $(LOCAL_PATH)/inference-engine/thirdparty/movidius/WinPthread \
	# $(LOCAL_PATH)/inference-engine/thirdparty/movidius/USB_WIN \

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
	-fno-exceptions \
	-frtti \
	-fexceptions

LOCAL_CFLAGS += \
	-DENABLE_MYRIAD=1 \
	-DENABLE_VPU \
	-DENABLE_OBJECT_DETECTION_TESTS=1 \
	-DENABLE_SEGMENTATION_TESTS=1 \
	-DHAVE_STRUCT_TIMESPEC \
	-DDEVICE_SHELL_ENABLED \
	-DUSE_USB_VSC \
	-D_CRT_SECURE_NO_WARNINGS \
	-D__PC__ \
	-D__ANDROID__ \
	-D_FORTIFY_SOURCE=2 \

LOCAL_SHARED_LIBRARIES := libusb liblog
LOCAL_STATIC_LIBRARIES := libXLink


include $(BUILD_SHARED_LIBRARY)
##########################################################################

include $(CLEAR_VARS)

LOCAL_MODULE := libXLink
LOCAL_PROPRIETARY_MODULE := true
# LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel


LOCAL_SRC_FILES := \
	inference-engine/thirdparty/movidius/XLink/pc/protocols/pcie_host.c \
	inference-engine/thirdparty/movidius/XLink/pc/protocols/usb_boot.c \
	inference-engine/thirdparty/movidius/XLink/pc/PlatformDeviceControl.c \
	inference-engine/thirdparty/movidius/XLink/pc/PlatformDeviceSearch.c \
	inference-engine/thirdparty/movidius/XLink/pc/PlatformData.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkPrivateFields.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkDispatcherImpl.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkDevice.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkData.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkStringUtils.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkPrivateDefines.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkDeprecated.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkStream.c \
	inference-engine/thirdparty/movidius/XLink/shared/src/XLinkDispatcher.c \

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/pc \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/pc/protocols \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/shared \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/shared/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/shared \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/shared/include \
	$(LIBUSB_HEADER) \
	$(LOCAL_PATH)/../../../../../external/libusb \
	$(LOCAL_PATH)/../../../../../external/libusb/libusb \

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
	-fno-exceptions \
	-frtti \
	-fexceptions

LOCAL_CFLAGS += \
	-DENABLE_MYRIAD=1 \
	-DENABLE_VPU \
	-DENABLE_OBJECT_DETECTION_TESTS=1 \
	-DENABLE_SEGMENTATION_TESTS=1 \
	-DHAVE_STRUCT_TIMESPEC \
	-DDEVICE_SHELL_ENABLED \
	-DUSE_USB_VSC \
	-D_CRT_SECURE_NO_WARNINGS \
	-D__PC__ \
	-D__ANDROID__ \
	-D_FORTIFY_SOURCE=2 \

LOCAL_SHARED_LIBRARIES := libusb liblog
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_STATIC_LIBRARY)

#########################################################################

####################################################
#include $(BUILD_STATIC_LIBRARY)
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2450.mvcmd
LOCAL_SRC_FILES := inference-engine/binary/vpu/firmware/ma2450/mvnc/MvNCAPI-ma2450.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
#####################################################
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2480.mvcmd
LOCAL_SRC_FILES := inference-engine/binary/vpu/firmware/ma2480/mvnc/MvNCAPI-ma2480.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
####################################################
#include $(PATH_TO_LIBUSB_SRC)/android/jni/libusb.mk
