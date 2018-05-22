LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := libgraphAPI
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_MULTILIB := both
#LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
    IRDocument.cpp \
    IRLayer.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
	$(LOCAL_PATH)/include \
	$(LOCAL_PATH)/../dl/inference-engine/include \
	$(LOCAL_PATH)/../dl/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/../dl/inference-engine/thirdparty/pugixml/src

LOCAL_CFLAGS += -std=c++11 -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DAKS -DENABLE_VPU -DENABLE_MYRIAD -DIMPLEMENT_INFERENCE_ENGINE_API
#LOCAL_CFLAGS += -DAKS -DNNLOG


LOCAL_SHARED_LIBRARIES := liblog
#		libinference_engine
#		libmyriadPlugin

include $(BUILD_STATIC_LIBRARY)
