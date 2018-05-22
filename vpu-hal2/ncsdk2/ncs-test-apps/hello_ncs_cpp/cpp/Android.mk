LOCAL_PATH:= $(call my-dir)

# ==================================

# executable: hello_ncs
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LIBUSB_HEADER:= $(LOCAL_PATH)/../../../../../../../../../../external/libusb/libusb

LOCAL_SRC_FILES := hello_ncs.cpp

LOCAL_MODULE := hello_ncs2

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
	$(LOCAL_PATH)/../../../../api/include \
	$(LIBUSB_HEADER) \
	$(LOCAL_PATH)/../../../../api/src/common/components/XLink/pc


LOCAL_CFLAGS += -O2 -Wall -pthread -fPIC -MMD -MP -fPIE

#LOCAL_SHARED_LIBRARIES := libmvnc
LOCAL_SHARED_LIBRARIES := libusb1.0 liblog libmvnc
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_EXECUTABLE)
