LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := libgraph_transformer
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
	inference-engine/src/vpu/graph_transformer/custom_layer.cpp \
	inference-engine/src/vpu/graph_transformer/graph_transformer_impl.cpp \
	inference-engine/src/vpu/graph_transformer/network_config.cpp \
	inference-engine/src/vpu/graph_transformer/simple_math.cpp \
	inference-engine/src/vpu/graph_transformer/hw/common.cpp \
	inference-engine/src/vpu/graph_transformer/hw/convolution.cpp \
	inference-engine/src/vpu/graph_transformer/hw/fc.cpp \
	inference-engine/src/vpu/graph_transformer/hw/fill_descriptors.cpp \
	inference-engine/src/vpu/graph_transformer/hw/inject_sw.cpp \
	inference-engine/src/vpu/graph_transformer/hw/pooling.cpp \
	inference-engine/src/vpu/graph_transformer/hw/split_depth_convolution.cpp \
	inference-engine/src/vpu/graph_transformer/hw/split_hw_descriptors.cpp \
	inference-engine/src/vpu/graph_transformer/hw/split_large_convolution.cpp \
	inference-engine/src/vpu/graph_transformer/hw/swap_concat_and_pool.cpp \
	inference-engine/src/vpu/graph_transformer/hw/try_hcw_layout.cpp \
	inference-engine/src/vpu/graph_transformer/ir/in_out_convert.cpp \
	inference-engine/src/vpu/graph_transformer/ir/parse_data.cpp \
	inference-engine/src/vpu/graph_transformer/ir/parse_network.cpp \
	inference-engine/src/vpu/graph_transformer/ir/pre_process.cpp \
	inference-engine/src/vpu/graph_transformer/optimizations/adjust_data_scale.cpp \
	inference-engine/src/vpu/graph_transformer/optimizations/allocate_resources.cpp \
	inference-engine/src/vpu/graph_transformer/optimizations/convert_order.cpp \
	inference-engine/src/vpu/graph_transformer/optimizations/eliminate_copy.cpp \
	inference-engine/src/vpu/graph_transformer/optimizations/eliminate_reshape.cpp \
	inference-engine/src/vpu/graph_transformer/optimizations/pack_postops.cpp \
	inference-engine/src/vpu/graph_transformer/stages/batch_norm.cpp \
	inference-engine/src/vpu/graph_transformer/stages/bias.cpp \
	inference-engine/src/vpu/graph_transformer/stages/clamp.cpp \
	inference-engine/src/vpu/graph_transformer/stages/concat.cpp \
	inference-engine/src/vpu/graph_transformer/stages/convolution.cpp \
	inference-engine/src/vpu/graph_transformer/stages/copy.cpp \
	inference-engine/src/vpu/graph_transformer/stages/crop.cpp \
	inference-engine/src/vpu/graph_transformer/stages/ctc_decoder.cpp \
	inference-engine/src/vpu/graph_transformer/stages/custom.cpp \
	inference-engine/src/vpu/graph_transformer/stages/deconvolution.cpp \
	inference-engine/src/vpu/graph_transformer/stages/detection_output.cpp \
	inference-engine/src/vpu/graph_transformer/stages/eltwise.cpp \
	inference-engine/src/vpu/graph_transformer/stages/elu.cpp \
	inference-engine/src/vpu/graph_transformer/stages/fc.cpp \
	inference-engine/src/vpu/graph_transformer/stages/flatten.cpp \
	inference-engine/src/vpu/graph_transformer/stages/grn.cpp \
	inference-engine/src/vpu/graph_transformer/stages/interp.cpp \
	inference-engine/src/vpu/graph_transformer/stages/mvn.cpp \
	inference-engine/src/vpu/graph_transformer/stages/none.cpp \
	inference-engine/src/vpu/graph_transformer/stages/normalize.cpp \
	inference-engine/src/vpu/graph_transformer/stages/norm.cpp \
	inference-engine/src/vpu/graph_transformer/stages/permute.cpp \
	inference-engine/src/vpu/graph_transformer/stages/pooling.cpp \
	inference-engine/src/vpu/graph_transformer/stages/power.cpp \
	inference-engine/src/vpu/graph_transformer/stages/prelu.cpp \
	inference-engine/src/vpu/graph_transformer/stages/priorbox_clustered.cpp \
	inference-engine/src/vpu/graph_transformer/stages/priorbox.cpp \
	inference-engine/src/vpu/graph_transformer/stages/proposal.cpp \
	inference-engine/src/vpu/graph_transformer/stages/psroipooling.cpp \
	inference-engine/src/vpu/graph_transformer/stages/region_yolo.cpp \
	inference-engine/src/vpu/graph_transformer/stages/relu.cpp \
	inference-engine/src/vpu/graph_transformer/stages/reorg_yolo.cpp \
	inference-engine/src/vpu/graph_transformer/stages/reshape.cpp \
	inference-engine/src/vpu/graph_transformer/stages/roipooling.cpp \
	inference-engine/src/vpu/graph_transformer/stages/scale.cpp \
	inference-engine/src/vpu/graph_transformer/stages/sigmoid.cpp \
	inference-engine/src/vpu/graph_transformer/stages/softmax.cpp \
	inference-engine/src/vpu/graph_transformer/stages/split.cpp \
	inference-engine/src/vpu/graph_transformer/stages/tanh.cpp \
	inference-engine/src/vpu/graph_transformer/stages/tile.cpp


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/include/details \
	$(LOCAL_PATH)/inference-engine/include/vpu \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
	$(LOCAL_PATH)/inference-engine/src/vpu/common \
	$(LOCAL_PATH)/inference-engine/src/vpu/myriad_plugin \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/hw \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/ir \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/optimizations \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/stages \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src


LOCAL_CFLAGS += -std=c++11 -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DENABLE_VPU -DENABLE_MYRIAD -D__ANDROID__ -DNDEBUG -DIMPLEMENT_INFERENCE_ENGINE_API -D_FORTIFY_SOURCE=2 -fPIE
#LOCAL_CFLAGS += -DNNLOG


LOCAL_STATIC_LIBRARIES := libvpu_common
LOCAL_SHARED_LIBRARIES := liblog

include $(BUILD_STATIC_LIBRARY)
##############################################
include $(CLEAR_VARS)

LOCAL_MODULE := libvpu_common
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
	inference-engine/src/vpu/common/vpu_logger.cpp \
	inference-engine/src/vpu/common/parsed_config.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/vpu/common \
	$(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer


LOCAL_CFLAGS += -std=c++11 -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -frtti -fexceptions -Wno-error
LOCAL_CFLAGS += -DENABLE_VPU -DENABLE_MYRIAD -D__ANDROID__ -DIMPLEMENT_INFERENCE_ENGINE_API -std=gnu++11 -D_FORTIFY_SOURCE=2 -fPIE
#LOCAL_CFLAGS += -DNNLOG

LOCAL_SHARED_LIBRARIES := libinference_engine

include $(BUILD_STATIC_LIBRARY)
