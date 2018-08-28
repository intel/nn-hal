LOCAL_PATH := $(my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := android.hardware.neuralnetworks@1.0-mkldnn-service
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.0-mkldnn.rc
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
LOCAL_SRC_FILES := \
    MklDnnDriver.cpp \
    MklDnnPreparedModel.cpp \
    service.cpp

LOCAL_CFLAGS += -Ivendor/intel/hardware/interfaces/mkl-dnn/include -fexceptions -Wno-unused-parameter

LOCAL_SHARED_LIBRARIES := \
    libhidlbase \
    libhidltransport \
    libhidlmemory \
    libutils \
    liblog \
    libcutils \
    libhardware \
    libbase \
    android.hidl.allocator@1.0 \
    android.hardware.neuralnetworks@1.0 \
    android.hidl.memory@1.0 \
    libmkldnn

LOCAL_MULTILIB := 64

include $(BUILD_EXECUTABLE)
