LOCAL_PATH := $(call my-dir)/../../../dldt

include $(CLEAR_VARS)

LOCAL_MODULE_OWNER := intel

LOCAL_MODULE := libclDNNPlugin

LOCAL_PROPRIETARY_MODULE := true

LOCAL_MULTILIB := 64

LOCAL_STATIC_LIBRARIES := libpugixml libomp

LOCAL_SHARED_LIBRARIES := liblog libinference_engine libclDNN64

LOCAL_CFLAGS += \
	-Wno-error \
	-Wall \
	-Wno-unknown-pragmas \
	-Wno-strict-overflow \
	-Wno-unused-variable \
	-Wno-unused-parameter \
	-Wno-non-virtual-dtor \
	-Wno-missing-field-initializers \
	-fstack-protector-all \
	-fexceptions \
	-frtti \
	-Wuninitialized \
	-Winit-self \
	-Wmaybe-uninitialized \
	-fPIE \
	-fPIC \
	-Wformat \
	-Wformat-security \
	-fstack-protector-strong \
	-O3 \
	-DNDEBUG \
	-D_FORTIFY_SOURCE=2 \
	-s \
	-fvisibility=hidden \
	-fvisibility=default \
	-fopenmp \
	-std=gnu++11 \
	-DENABLE_CLDNN=1 \
	-DENABLE_GNA \
	-DENABLE_MKL_DNN=1 \
	-DENABLE_OBJECT_DETECTION_TESTS=1 \
	-DENABLE_PROFILING_ITT=0 \
	-DENABLE_SEGMENTATION_TESTS=1 \
	-DIE_BUILD_POSTFIX=\"\" \
	-DIE_THREAD=IE_THREAD_OMP \
	-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN \
	-DclDNNPlugin_EXPORTS \
	-D__ANDROID__ -DNNLOG -DDEBUG

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/cldnn_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/thirdparty/clDNN/api \
	$(LOCAL_PATH)/inference-engine/thirdparty/clDNN/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src

LOCAL_SRC_FILES += \
	inference-engine/src/cldnn_engine/dllmain.cpp \
	inference-engine/src/cldnn_engine/cldnn_custom_layer.cpp \
	inference-engine/src/cldnn_engine/cldnn_engine.cpp \
	inference-engine/src/cldnn_engine/cldnn_graph.cpp \
	inference-engine/src/cldnn_engine/cldnn_infer_request.cpp \
	inference-engine/src/cldnn_engine/debug_options.cpp \
	inference-engine/src/cldnn_engine/simple_math.cpp

include $(BUILD_SHARED_LIBRARY)
