LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libMKLDNNPlugin
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MULTILIB := both
LOCAL_MODULE_OWNER := intel

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/include

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/src/mkldnn_plugin \
	$(LOCAL_PATH)/inference-engine/src/mkldnn_plugin/mkldnn \
	$(LOCAL_PATH)/inference-engine/src/mkldnn_plugin/nodes

LOCAL_CFLAGS += \
	-std=c++11 \
	-fvisibility=default \
	-fPIC \
	-fPIE \
	-fstack-protector-all \
	-fexceptions \
	-O3 \
	-frtti \
	-fopenmp \
	-Wno-error \
	-Wall \
	-Wno-unknown-pragmas \
	-Wno-strict-overflow \
	-Wformat \
	-Wformat-security \
	-Wno-unused-variable \
	-Wno-unused-parameter \
	-Wno-non-virtual-dtor \
	-Wno-missing-field-initializers \
	-D_FORTIFY_SOURCE=2

LOCAL_CFLAGS += \
	-DENABLE_MKL_DNN \
	-DMKLDNN_OPENMP \
	-DMKL_VERSION=\"v0.15_beta\" \
	-DMKLDNN_DEPRECATED_ROI \
	-DNDEBUG \
	-DAKS -DNNLOG -DDEBUG \
	-DIMPLEMENT_INFERENCE_ENGINE_API

LOCAL_STATIC_LIBRARIES := libomp
LOCAL_SHARED_LIBRARIES := liblog libinference_engine libmkldnn

LOCAL_SRC_FILES += \
	inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_omp_manager.cpp

LOCAL_SRC_FILES += \
	inference-engine/src/mkldnn_plugin/config.cpp \
	inference-engine/src/mkldnn_plugin/mean_image.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_async_infer_request.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_descriptor.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_extension_utils.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_graph.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_graph_optimizer.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_memory.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_node.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn_plugin.cpp \
	inference-engine/src/mkldnn_plugin/mkldnn/iml_type_mapper.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_activation_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_batchnorm_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_clamp_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_concat_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_conv_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_crop_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_eltwise_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_fullyconnected_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_generic_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_input_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_lrn_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_memory_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_permute_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_pooling_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_power_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_reorder_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_reshape_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_scaleshift_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp \
	inference-engine/src/mkldnn_plugin/nodes/mkldnn_tile_node.cpp

include $(BUILD_SHARED_LIBRARY)
