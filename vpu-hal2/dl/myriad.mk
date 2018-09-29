LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := libmyriadPlugin
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MULTILIB := both
#LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/src/vpu/myriad_plugin/myriad_async_infer_request.cpp \
	inference-engine/src/vpu/myriad_plugin/myriad_executor.cpp \
	inference-engine/src/vpu/myriad_plugin/myriad_infer_request.cpp \
	inference-engine/src/vpu/myriad_plugin/myriad_plugin.cpp


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/vpu \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/src/vpu/myriad_plugin \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
	$(LOCAL_PATH)/inference-engine/temp/myriad/include \
	$(LOCAL_PATH)/inference-engine/src/vpu/common


LOCAL_CFLAGS += -std=c++11 -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DENABLE_VPU -DENABLE_MYRIAD -DAKS -DIMPLEMENT_INFERENCE_ENGINE_API -fvisibility=default -std=gnu++11 -D_FORTIFY_SOURCE=2 -fPIE
#LOCAL_CFLAGS += -DAKS

LOCAL_STATIC_LIBRARIES := libgraph_transformer libvpu_common

LOCAL_SHARED_LIBRARIES := libmvnc libinference_engine liblog

include $(BUILD_SHARED_LIBRARY)
