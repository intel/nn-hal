LOCAL_PATH := $(call my-dir)

#include $(CLEAR_VARS)
#include $(LOCAL_PATH)/dl/Android.mk
##############################################################
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := android.hardware.neuralnetworks@1.0-vpu-impl
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/lib
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
    VpuDriver.cpp \
    VpuPreparedModel.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
	$(LOCAL_PATH)/graphAPI \
	$(LOCAL_PATH)/dl \
	$(LOCAL_PATH)/dl/inference-engine/thirdparty/pugixml/src \
  $(LOCAL_PATH)/dl/inference-engine/include \
	$(LOCAL_PATH)/dl/inference-engine/include/cpp \
	$(LOCAL_PATH)/dl/inference-engine/include/details \
	$(LOCAL_PATH)/dl/inference-engine/include/details/os \
	$(LOCAL_PATH)/dl/inference-engine/include/cldnn \
	$(LOCAL_PATH)/dl/inference-engine/include/gna \
	$(LOCAL_PATH)/dl/inference-engine/include/hetero \
	$(LOCAL_PATH)/dl/inference-engine/include/mkldnn \
	$(LOCAL_PATH)/dl/inference-engine/include/openvx \
	$(LOCAL_PATH)/dl/inference-engine/include/vpu \
  $(LOCAL_PATH)/dl/inference-engine/src/inference_engine \
  $(LOCAL_PATH)/dl/inference-engine/src/dumper \
  $(LOCAL_PATH)/dl/inference-engine/src/inference_engine/cpp_interfaces \
  $(LOCAL_PATH)/dl/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/dl/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/dl/inference-engine/src/inference_engine/cpp_interfaces/interface


LOCAL_CFLAGS += -std=c++11 -Wall -Wno-unused-variable -Wno-unused-parameter -fPIC -Wno-non-virtual-dtor -Wno-missing-field-initializers  -Wno-error -fexceptions

LOCAL_CFLAGS += -DENABLE_VPU -DAKS -DENABLE_MYRIAD -DIMPLEMENT_INFERENCE_ENGINE_API -fvisibility=default -D_FORTIFY_SOURCE=2 -fPIE
#LOCAL_CFLAGS += -DAKS -DNNLOG
#LOCAL_CFLAGS += -DVPU_DEBUG

LOCAL_SHARED_LIBRARIES := \
    libhidlbase \
    libhidltransport \
    libutils \
    liblog \
    libcutils \
    libhardware \
    libbase \
    libhidlmemory \
    android.hardware.neuralnetworks@1.0 \
    android.hidl.allocator@1.0 \
    android.hidl.memory@1.0 \
    libinference_engine


LOCAL_STATIC_LIBRARIES := libgraphAPI libpugixml

include $(BUILD_SHARED_LIBRARY)
###############################################################
#$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)
LOCAL_MODULE := android.hardware.neuralnetworks@1.0-vpu-service
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.0-vpu.rc
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_SRC_FILES := \
    service.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)


LOCAL_CFLAGS += -Ivendor/intel/hardware/interfaces/ml/vpu-hal -fexceptions -fPIE

LOCAL_SHARED_LIBRARIES := \
    libhidlbase \
    libhidltransport \
    libutils \
    liblog \
    libcutils \
    libhardware \
    android.hardware.neuralnetworks@1.0 \
		android.hardware.neuralnetworks@1.0-vpu-impl


#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

include $(BUILD_EXECUTABLE)
#############################################################

ZPATH := $(LOCAL_PATH)
include $(CLEAR_VARS)

include $(ZPATH)/graphAPI/graphAPI.mk
include $(ZPATH)/graphTests/graphTests.mk
include $(ZPATH)/ncsdk2/api/src/Android.mk
include $(ZPATH)/dl/Android.mk

PATH_TO_LIBUSB_SRC:= $(ZPATH)/../../../../../../external/libusb
include $(PATH_TO_LIBUSB_SRC)/android/jni/libusb.mk
