LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := graphtest_vpu
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	main.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
	$(LOCAL_PATH)/../graphAPI \
	$(LOCAL_PATH)/../dl/inference-engine/include \
	$(LOCAL_PATH)/../dl/inference-engine/include/cpp \
	$(LOCAL_PATH)/../dl/inference-engine/include/details \
	$(LOCAL_PATH)/../dl/inference-engine/include/vpu \
	$(LOCAL_PATH)/../dl/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/../dl/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/../dl/inference-engine/thirdparty/pugixml/src

LOCAL_CFLAGS += \
	-std=c++11 \
	-Wall \
	-Wno-unknown-pragmas \
	-Wno-strict-overflow \
	-fPIC \
	-fPIE \
	-Wformat \
	-Wformat-security \
	-fstack-protector-all \
	-Wno-unused-variable \
	-Wno-unused-parameter \
	-Wno-non-virtual-dtor \
	-Wno-missing-field-initializers \
	-fexceptions \
	-frtti \
	-Wno-error \
	-D_FORTIFY_SOURCE=2

LOCAL_CFLAGS += \
	-DNNLOG \
	-DENABLE_MYRIAD \
	-DAKS \
	-DIMPLEMENT_INFERENCE_ENGINE_API

LOCAL_STATIC_LIBRARIES := libgraphAPI libpugixml
LOCAL_SHARED_LIBRARIES := libinference_engine liblog

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_MODULE := graphtest_cpu
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	main.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
	$(LOCAL_PATH)/../graphAPI \
	$(LOCAL_PATH)/../dl/inference-engine/include \
	$(LOCAL_PATH)/../dl/inference-engine/include/cpp \
	$(LOCAL_PATH)/../dl/inference-engine/include/details \
	$(LOCAL_PATH)/../dl/inference-engine/include/vpu \
	$(LOCAL_PATH)/../dl/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/../dl/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/../dl/inference-engine/thirdparty/pugixml/src

LOCAL_CFLAGS += \
	-std=c++11 \
	-Wall \
	-Wno-unknown-pragmas \
	-Wno-strict-overflow \
	-fPIC \
	-fPIE \
	-Wformat \
	-Wformat-security \
	-fstack-protector-all \
	-Wno-unused-variable \
	-Wno-unused-parameter \
	-Wno-non-virtual-dtor \
	-Wno-missing-field-initializers \
	-fexceptions \
	-frtti \
	-Wno-error \
	-D_FORTIFY_SOURCE=2

LOCAL_CFLAGS += \
	-DNNLOG \
	-DENABLE_MKLDNN \
	-DAKS \
	-DIMPLEMENT_INFERENCE_ENGINE_API

LOCAL_STATIC_LIBRARIES := libgraphAPI libpugixml
LOCAL_SHARED_LIBRARIES := libinference_engine liblog

include $(BUILD_EXECUTABLE)
