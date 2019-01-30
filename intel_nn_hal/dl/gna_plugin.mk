LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := libGNAPlugin
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/src/gna_plugin/dnn.cpp \
	inference-engine/src/gna_plugin/dnn_memory.cpp \
	inference-engine/src/gna_plugin/floatmath.cpp \
	inference-engine/src/gna_plugin/gna_device.cpp \
	inference-engine/src/gna_plugin/gna_helper.cpp \
	inference-engine/src/gna_plugin/gna_model_serial.cpp \
	inference-engine/src/gna_plugin/gna_plugin.cpp \
	inference-engine/src/gna_plugin/gna_plugin_entry_points.cpp \
	inference-engine/src/gna_plugin/gna_plugin_passes.cpp \
	inference-engine/src/gna_plugin/lstm.cpp \
	inference-engine/src/gna_plugin/pwl_design.cpp \
	inference-engine/src/gna_plugin/util.cpp \
	inference-engine/src/gna_plugin/pwl_design.cpp \
	inference-engine/src/gna_plugin/quantization/quantization.cpp



LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/vpu \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/src/gna_plugin \
	$(LOCAL_PATH)/inference-engine/src/gna_plugin/quantization \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
	$(LOCAL_PATH)/inference-engine/temp/gna/linux/include


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
	-Wno-error

LOCAL_CFLAGS += \
	-DENABLE_GNA=1 \
	-DENABLE_MYRIAD=1 \
	-DENABLE_VPU \
	-DIMPLEMENT_INFERENCE_ENGINE_API \
	-D_FORTIFY_SOURCE=2 \
	-DNDEBUG \
	-DENABLE_OBJECT_DETECTION_TESTS=1 \
	-DGNAPlugin_EXPORTS \
	-D__ANDROID__ \
	-DNNLOG


LOCAL_STATIC_LIBRARIES :=

LOCAL_SHARED_LIBRARIES := libgna_api libgna_kernel libinference_engine liblog

include $(BUILD_SHARED_LIBRARY)

##################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libGNAxx
#LOCAL_SRC_FILES := inference-engine/temp/myriad/lib/libmvnc.so
LOCAL_SRC_FILES := lib/libgna.so
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_SUFFIX := .so
#LOCAL_MODULE_CLASS := SHARED_LIBRARIES

#include $(BUILD_PREBUILT)
include $(PREBUILT_SHARED_LIBRARY)
