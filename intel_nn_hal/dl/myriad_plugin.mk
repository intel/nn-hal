LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libmyriadPlugin
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/src/vpu/myriad_plugin/myriad_async_infer_request.cpp \
	inference-engine/src/vpu/myriad_plugin/myriad_config.cpp \
	inference-engine/src/vpu/myriad_plugin/myriad_executable_network.cpp \
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
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/allocator \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/backend \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/frontend \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/hw \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/model \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/sw \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include/vpu/utils \


LOCAL_CFLAGS += \
	-std=c++11 \
	-fvisibility=default \
	-Wall \
	-Wno-unknown-pragmas \
	-Wno-strict-overflow \
	-O3 \
	-fPIE \
	-fPIC \
	-frtti \
	-Wformat \
	-Wformat-security \
	-fstack-protector-all \
	-Wno-unused-variable \
	-Wno-unused-parameter \
	-Wno-non-virtual-dtor \
	-Wno-missing-field-initializers \
	-fexceptions \
	-Wno-error \
	-pthread \

LOCAL_CFLAGS += \
	-DENABLE_MYRIAD \
	-DENABLE_VPU \
	-DENABLE_GNA \
	-DIMPLEMENT_INFERENCE_ENGINE_API \
	-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN \
	-D_FORTIFY_SOURCE=2 \
	-DNDEBUG \
	-DENABLE_OBJECT_DETECTION_TESTS=1 \
	-DmyriadPlugin_EXPORTS \
	-D__ANDROID__ \
	-DCI_BUILD_NUMBER='""' \
#	-DNNLOG

#LOCAL_CFLAGS += -D__ANDROID__ -DNNLOG

LOCAL_STATIC_LIBRARIES := libvpu_graph_transformer

LOCAL_SHARED_LIBRARIES := libmvnc libinference_engine liblog

include $(BUILD_SHARED_LIBRARY)

##################################################
