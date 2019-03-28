LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libinference_engine
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
inference-engine/src/inference_engine/precision_utils.cpp \
inference-engine/src/inference_engine/ie_blob_common.cpp \
inference-engine/src/inference_engine/ie_utils.cpp \
inference-engine/src/inference_engine/ade_util.cpp \
inference-engine/src/inference_engine/cpu_detector.cpp \
inference-engine/src/inference_engine/network_serializer.cpp \
inference-engine/src/inference_engine/ie_layers.cpp \
inference-engine/src/inference_engine/ie_device.cpp \
inference-engine/src/inference_engine/ie_network.cpp \
inference-engine/src/inference_engine/ie_util_internal.cpp \
inference-engine/src/inference_engine/cnn_network_stats_impl.cpp \
inference-engine/src/inference_engine/ie_format_parser.cpp \
inference-engine/src/inference_engine/ie_layouts.cpp \
inference-engine/src/inference_engine/ie_graph_splitter.cpp \
inference-engine/src/inference_engine/ie_context.cpp \
inference-engine/src/inference_engine/ie_preprocess_data.cpp \
inference-engine/src/inference_engine/blob_factory.cpp \
inference-engine/src/inference_engine/ie_data.cpp \
inference-engine/src/inference_engine/net_pass.cpp \
inference-engine/src/inference_engine/xml_parse_utils.cpp \
inference-engine/src/inference_engine/ie_layers_internal.cpp \
inference-engine/src/inference_engine/ie_layer_parsers.cpp \
inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp \
inference-engine/src/inference_engine/data_stats.cpp \
inference-engine/src/inference_engine/graph_transformer.cpp \
inference-engine/src/inference_engine/memory_solver.cpp \
inference-engine/src/inference_engine/graph_tools.cpp \
inference-engine/src/inference_engine/ie_layer_validators.cpp \
inference-engine/src/inference_engine/ie_memcpy.cpp \
inference-engine/src/inference_engine/file_utils.cpp \
inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp \
inference-engine/src/inference_engine/ie_version.cpp \
inference-engine/src/inference_engine/cnn_network_impl.cpp \
inference-engine/src/inference_engine/system_alllocator.cpp \
inference-engine/src/inference_engine/blob_transform.cpp \
inference-engine/src/inference_engine/builders/ie_crop_layer.cpp \
inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp \
inference-engine/src/inference_engine/builders/ie_simpler_nms_layer.cpp \
inference-engine/src/inference_engine/builders/ie_split_layer.cpp \
inference-engine/src/inference_engine/builders/ie_prior_box_layer.cpp \
inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp \
inference-engine/src/inference_engine/builders/ie_eltwise_layer.cpp \
inference-engine/src/inference_engine/builders/ie_pooling_layer.cpp \
inference-engine/src/inference_engine/builders/ie_ctc_greedy_decoder_layer.cpp \
inference-engine/src/inference_engine/builders/ie_power_layer.cpp \
inference-engine/src/inference_engine/builders/ie_concat_layer.cpp \
inference-engine/src/inference_engine/builders/ie_psroi_pooling_layer.cpp \
inference-engine/src/inference_engine/builders/ie_const_layer.cpp \
inference-engine/src/inference_engine/builders/ie_layer_builder.cpp \
inference-engine/src/inference_engine/builders/ie_elu_layer.cpp \
inference-engine/src/inference_engine/builders/ie_prior_box_clustered_layer.cpp \
inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp \
inference-engine/src/inference_engine/builders/ie_grn_layer.cpp \
inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp \
inference-engine/src/inference_engine/builders/ie_tile_layer.cpp \
inference-engine/src/inference_engine/builders/ie_region_yolo_layer.cpp \
inference-engine/src/inference_engine/builders/ie_reshape_layer.cpp \
inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp \
inference-engine/src/inference_engine/builders/ie_normalize_layer.cpp \
inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp \
inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp \
inference-engine/src/inference_engine/builders/ie_clamp_layer.cpp \
inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp \
inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp \
inference-engine/src/inference_engine/builders/ie_network_builder.cpp \
inference-engine/src/inference_engine/builders/ie_norm_layer.cpp \
inference-engine/src/inference_engine/builders/ie_memory_layer.cpp \
inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp \
inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp \
inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp \
inference-engine/src/inference_engine/builders/ie_sigmoid_layer.cpp \
inference-engine/src/inference_engine/builders/ie_permute_layer.cpp \
inference-engine/src/inference_engine/builders/ie_relu_layer.cpp \
inference-engine/src/inference_engine/builders/ie_layer_fragment.cpp \
inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp \
inference-engine/src/inference_engine/builders/ie_convolution_layer.cpp \
inference-engine/src/inference_engine/builders/ie_detection_output_layer.cpp \
inference-engine/src/inference_engine/builders/ie_scale_shift_layer.cpp \
inference-engine/src/inference_engine/builders/ie_prelu_layer.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_task_with_stages.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_task.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp \
inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp \
inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp \
inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp \
inference-engine/src/inference_engine/cpu_x86_sse42/ie_preprocess_data_sse42.cpp \
inference-engine/src/inference_engine/cpu_x86_sse42/blob_transform_sse42.cpp \
inference-engine/src/inference_engine/shape_infer/built-in/ie_built_in_holder.cpp \




LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/cldnn \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/include/details \
	$(LOCAL_PATH)/inference-engine/include/details/os \
	$(LOCAL_PATH)/inference-engine/include/dlia \
	$(LOCAL_PATH)/inference-engine/include/gna \
	$(LOCAL_PATH)/inference-engine/include/hetero \
	$(LOCAL_PATH)/inference-engine/include/openvx \
	$(LOCAL_PATH)/inference-engine/include/vpu \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/dumper \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/builder \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpu_x86_sse42 \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/shape_infer \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/shape_infer/built-in \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/../ade/sources/ade/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu/xbyak \
	$(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include
	#$(LOCAL_PATH)/../../../../../../../external/clang/lib/Headers


LOCAL_CFLAGS += -DIE_THREAD=IE_THREAD_OMP -DIMPLEMENT_INFERENCE_ENGINE_API -DGAPI_STANDALONE -std=c++11  -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all -msse4.2
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DENABLE_MKL_DNN -D__ANDROID__ -DNDEBUG -DIMPLEMENT_INFERENCE_ENGINE_API -fvisibility=default -std=gnu++11 -D_FORTIFY_SOURCE=2 -fPIE -DUSE_STATIC_IE
#LOCAL_CFLAGS += -DNNLOG



#Note: check for sse compile flag in android
#include for sse4 headers -> external/clang/lib/Headers/nmmintrin.h
#set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/blob_transform_sse42.cpp PROPERTIES COMPILE_FLAGS -msse4.2)
#set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_data_sse42.cpp PROPERTIES COMPILE_FLAGS -msse4.2)

LOCAL_SHARED_LIBRARIES := liblog
LOCAL_STATIC_LIBRARIES := libpugixml libade

include $(BUILD_SHARED_LIBRARY)
##########################################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libpugixml
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/pugixml/src/pugixml.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/src/inference_engine


LOCAL_CFLAGS += -std=c++11  -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DNDEBUG -D__ANDROID__

LOCAL_SHARED_LIBRARIES :=
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_STATIC_LIBRARY)

##########################################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libade
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MULTILIB := both
#LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	../ade/sources/ade/source/alloc.cpp \
	../ade/sources/ade/source/assert.cpp \
	../ade/sources/ade/source/check_cycles.cpp \
	../ade/sources/ade/source/edge.cpp \
	../ade/sources/ade/source/execution_engine.cpp \
	../ade/sources/ade/source/graph.cpp \
	../ade/sources/ade/source/memory_accessor.cpp \
	../ade/sources/ade/source/memory_descriptor.cpp \
	../ade/sources/ade/source/memory_descriptor_ref.cpp \
	../ade/sources/ade/source/memory_descriptor_view.cpp \
	../ade/sources/ade/source/metadata.cpp \
	../ade/sources/ade/source/metatypes.cpp \
	../ade/sources/ade/source/node.cpp \
	../ade/sources/ade/source/search.cpp \
	../ade/sources/ade/source/subgraphs.cpp \
	../ade/sources/ade/source/topological_sort.cpp \
	../ade/sources/ade/source/passes/communications.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/../ade/sources/ade/include \
	$(LOCAL_PATH)/..//ade/sources/ade/include/ade \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/communication \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/execution_engine \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/helpers \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/memory \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/metatypes \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/passes \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/util


LOCAL_CFLAGS += -std=c++11  -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DNDEBUG -D__ANDROID__

LOCAL_SHARED_LIBRARIES :=
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_STATIC_LIBRARY)
