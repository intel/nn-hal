
LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_SRC_FILES := fp.cpp vpu_lib.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../ncsdk/include \
                    $(LOCAL_PATH)/../graph_compiler_NCS \
                    $(LOCAL_PATH)
LOCAL_SHARED_LIBRARIES := libmvnc liblog libutils
LOCAL_CPPFLAGS := -fexceptions -o3
LOCAL_MODULE := libncs_nn_operation


include $(BUILD_SHARED_LIBRARY)
