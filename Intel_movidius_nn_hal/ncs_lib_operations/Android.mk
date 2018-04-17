
LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_SRC_FILES := fp.cpp vpu_lib.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../ncsdk/include \
                    $(LOCAL_PATH)/../graph_compiler_NCS \
                    $(LOCAL_PATH)
LOCAL_SHARED_LIBRARIES := libmvnc liblog libutils
LOCAL_CPPFLAGS := -fexceptions
LOCAL_MODULE := libncs_nn_operation


include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := graph_sigm
LOCAL_MODULE_TAGS := optional
LOCAL_MODULE_CLASS := ETC
LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc
LOCAL_SRC_FILES := graphfiles/graph_sigm
include $(BUILD_PREBUILT)

include $(CLEAR_VARS)
LOCAL_MODULE := graph_tanh
LOCAL_MODULE_TAGS := optional
LOCAL_MODULE_CLASS := ETC
LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc
LOCAL_SRC_FILES := graphfiles/graph_tanh
include $(BUILD_PREBUILT)
