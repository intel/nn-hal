LOCAL_PATH := $(call my-dir)/../../../dldt
#LOCAL_PATH := $(call my-dir)

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
	inference-engine/src/gna_plugin/lstm.cpp \
	inference-engine/src/gna_plugin/pwl_design.cpp \
	inference-engine/src/gna_plugin/util.cpp \
	inference-engine/src/gna_plugin/quantization/quantization.cpp \
	inference-engine/src/gna_plugin/gna_plugin_passes.cpp


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/vpu \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/include/gna \
	$(LOCAL_PATH)/inference-engine/src/gna_plugin \
	$(LOCAL_PATH)/inference-engine/src/gna_plugin/quantization \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpu_x86_sse42 \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/shape_infer \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/shape_infer/built-in \
	$(LOCAL_PATH)/inference-engine/temp/gna/linux/include \
	$(LOCAL_PATH)/inference-engine/include/cldnn \
	$(LOCAL_PATH)/inference-engine/include/details \
	$(LOCAL_PATH)/inference-engine/include/details/os \
	$(LOCAL_PATH)/inference-engine/include/dlia \
	$(LOCAL_PATH)/inference-engine/include/hetero \
	$(LOCAL_PATH)/inference-engine/include/openvx \
	$(LOCAL_PATH)/inference-engine/src/dumper \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/sources/ade/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu/xbyak


LOCAL_CFLAGS += \
	-std=c++11 \
	-g \
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
    -Wno-c++11-narrowing \
	-pthread \
	-msse4.2

LOCAL_CFLAGS += \
	-DENABLE_GNA=1 \
	-DIMPLEMENT_INFERENCE_ENGINE_API \
	-D_FORTIFY_SOURCE=2 \
	-DNDEBUG \
	-DENABLE_OBJECT_DETECTION_TESTS=1 \
	-DGNAPlugin_EXPORTS \
	-D__ANDROID__ \
	-DNNLOG

LOCAL_CFLAGS += \
    -D__GNA_ANDROID__ \
	-D_NO_MKL_ \
	-std=c++11


LOCAL_STATIC_LIBRARIES :=

LOCAL_SHARED_LIBRARIES := libgna libinference_engine liblog

include $(BUILD_SHARED_LIBRARY)

##################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libgna
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := gna_lib/libgna.so
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_SUFFIX := .so
LOCAL_MODULE_CLASS := SHARED_LIBRARIES
LOCAL_SHARED_LIBRARIES := libc++_shared

include $(BUILD_PREBUILT)

##################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libc++_shared
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := gna_lib/libc++_shared.so
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_SUFFIX := .so
LOCAL_MODULE_CLASS := SHARED_LIBRARIES

include $(BUILD_PREBUILT)
