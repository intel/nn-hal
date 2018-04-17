LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := android.hardware.neuralnetworks@1.0-vpudriver-impl
LOCAL_PROPRIETARY_MODULE := true
LOCAL_SRC_FILES := \
    src/vpu_driver/VpuDriver.cpp \
    src/vpu_driver/VpuPreparedModel.cpp \
		src/vpu_driver/VpuExecutor.cpp \
    src/vpu_driver/VpuUtils.cpp \
		src/vpu_operations/VpuOperationsUtils.cpp \
		src/vpu_operations/VpuActivation.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
  $(LOCAL_PATH)/include \
  $(LOCAL_PATH)/../ncsdk/include \
  $(LOCAL_PATH)/../ncs_lib_operations \
	$(LOCAL_PATH)/../graph_compiler_NCS \
	frameworks/ml/nn/runtime/include


#TODO update the CFLAGS

LOCAL_CFLAGS += -fexceptions

LOCAL_SHARED_LIBRARIES := \
                    libhidlbase \
                    libhidltransport \
                    libutils \
                    liblog \
                    libcutils \
                    libhardware \
                    libbase \
                    libcutils \
                    libhidlmemory \
                    android.hardware.neuralnetworks@1.0 \
                    android.hidl.allocator@1.0 \
                    libmvnc \
                    libncs_nn_operation \
										libncs_graph_compiler \
                    android.hidl.memory@1.0

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := android.hardware.neuralnetworks@1.0-service-vpudriver
LOCAL_INIT_RC := android.hardware.neuralnetworks@1.0-vpudriver.rc
LOCAL_MODULE_RELATIVE_PATH := hw
LOCAL_PROPRIETARY_MODULE := true
LOCAL_SRC_FILES := service.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH) \
  $(LOCAL_PATH)/include \
  $(LOCAL_PATH)/../ncsdk/include \
  $(LOCAL_PATH)/../ncs_lib_operations \
	frameworks/ml/nn/runtime/include

LOCAL_CFLAGS += -fexceptions

LOCAL_SHARED_LIBRARIES := \
            libhidlbase \
            libhidltransport \
            libutils \
            liblog \
            libcutils \
            libhardware \
            libhidlmemory \
            android.hardware.neuralnetworks@1.0 \
            android.hidl.allocator@1.0 \
            android.hidl.memory@1.0 \
            android.hardware.neuralnetworks@1.0-vpudriver-impl

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)
LOCAL_MODULE := app_ncs_test1
LOCAL_SRC_FILES := tests/ncs_test/ncs_test1.cpp

LOCAL_C_INCLUDES +=  \
                 $(LOCAL_PATH) \
                 $(LOCAL_PATH)/../ncsdk/include

LOCAL_CFLAGS += -fexceptions

LOCAL_SHARED_LIBRARIES := \
                    libhidlbase \
                    libhidltransport \
                    libutils \
                    liblog \
                    libcutils \
                    libhardware \
                    libbase \
                    libcutils \
                    libhidlmemory \
                    libmvnc


include $(BUILD_EXECUTABLE)
