
LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
LOCAL_SRC_FILES := Blob.cpp \
                   android_stage_dummy.cpp \
                   input_stage.cpp \
									 stage_logistic.cpp \
									 stage_relu.cpp \
									 stage_conv2D.cpp \
									 stage_depthconv2D.cpp \
									 stage_pooling.cpp \
									 stage_softmax.cpp \
									 stage_reshape.cpp \
									 stage_tanh.cpp

LOCAL_C_INCLUDES += $(LOCAL_PATH) \
										$(LOCAL_PATH)/../ncs_lib_operations

LOCAL_SHARED_LIBRARIES := libmvnc libncs_nn_operation liblog libutils
LOCAL_CPPFLAGS := -fexceptions
LOCAL_MODULE := libncs_graph_compiler


include $(BUILD_SHARED_LIBRARY)
