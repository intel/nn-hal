LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := libgraphAPI
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	IRDocument.cpp \
	IRLayer.cpp \
	IENetwork.cpp \
    IRBuilder.cpp \
	GnaNetwork.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
	$(LOCAL_PATH)/../../../dldt/inference-engine/include \
	$(LOCAL_PATH)/../../../dldt/inference-engine/include/builders \
	$(LOCAL_PATH)/../../../dldt/inference-engine/include/cpp \
	$(LOCAL_PATH)/../../../dldt/inference-engine/include/details \
	$(LOCAL_PATH)/../../../dldt/inference-engine/include/gna \
	$(LOCAL_PATH)/../../../dldt/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/../../../dldt/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/../../../dldt/inference-engine/thirdparty/pugixml/src \

LOCAL_CFLAGS += \
	-std=c++17 \
	-fPIC \
	-fPIE \
	-Wall \
	-Wno-unused-variable \
	-Wno-unused-parameter \
	-Wno-non-virtual-dtor \
	-Wno-missing-field-initializers \
	-frtti \
	-Wno-error \
	-D_FORTIFY_SOURCE=2 \
	-fvisibility=default \
	-fexceptions

LOCAL_CFLAGS += \
	-D__ANDROID__ \
	-DIMPLEMENT_INFERENCE_ENGINE_API \
	-DENABLE_GNA
#	-DNNLOG \

LOCAL_SHARED_LIBRARIES := liblog \
			android.hardware.neuralnetworks@1.2 \
			libhardware

include $(BUILD_STATIC_LIBRARY)
