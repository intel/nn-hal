LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := libinference_engine
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MULTILIB := both
#LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	inference-engine/src/inference_engine/ie_layers.cpp \
	inference-engine/src/inference_engine/ade_util.cpp \
	inference-engine/src/inference_engine/blob_factory.cpp \
	inference-engine/src/inference_engine/cnn_network_impl.cpp \
	inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp \
	inference-engine/src/inference_engine/cpp_interfaces/ie_task.cpp \
	inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp \
	inference-engine/src/inference_engine/cpp_interfaces/ie_task_with_stages.cpp \
	inference-engine/src/inference_engine/file_utils.cpp \
	inference-engine/src/inference_engine/graph_transformer.cpp \
	inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp \
	inference-engine/src/inference_engine/ie_data.cpp \
	inference-engine/src/inference_engine/ie_device.cpp \
	inference-engine/src/inference_engine/ie_graph_splitter.cpp \
	inference-engine/src/inference_engine/ie_layouts.cpp \
	inference-engine/src/inference_engine/ie_util_internal.cpp \
	inference-engine/src/inference_engine/ie_utils.cpp \
	inference-engine/src/inference_engine/ie_version.cpp \
	inference-engine/src/inference_engine/precision_utils.cpp \
	inference-engine/src/inference_engine/system_alllocator.cpp \
	inference-engine/src/inference_engine/v2_format_parser.cpp \
	inference-engine/src/inference_engine/xml_parse_utils.cpp \
	inference-engine/src/inference_engine/ie_global.cpp


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/include/cpp \
	$(LOCAL_PATH)/inference-engine/include/vpu \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/dumper \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
	$(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/common/include


LOCAL_CFLAGS += -std=c++11  -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DENABLE_VPU -DENABLE_MYRIAD -DAKS -DNDEBUG -DIMPLEMENT_INFERENCE_ENGINE_API -fvisibility=default -std=gnu++11 -D_FORTIFY_SOURCE=2 -fPIE -DUSE_STATIC_IE
#LOCAL_CFLAGS += -DAKS -DNNLOG
LOCAL_CFLAGS += -DENABLE_MKL_DNN

LOCAL_SHARED_LIBRARIES := liblog
LOCAL_STATIC_LIBRARIES := libpugixml libade

include $(BUILD_SHARED_LIBRARY)
##########################################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libpugixml
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MULTILIB := both
#LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/pugixml/src/pugixml.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
	$(LOCAL_PATH)/inference-engine/src/inference_engine


LOCAL_CFLAGS += -std=c++11  -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DENABLE_VPU -DENABLE_MYRIAD -DNDEBUG -DAKS

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
	inference-engine/thirdparty/ade/ade/source/alloc.cpp \
	inference-engine/thirdparty/ade/ade/source/check_cycles.cpp \
	inference-engine/thirdparty/ade/ade/source/edge.cpp \
	inference-engine/thirdparty/ade/ade/source/execution_engine.cpp \
	inference-engine/thirdparty/ade/ade/source/graph.cpp \
	inference-engine/thirdparty/ade/ade/source/memory_accessor.cpp \
	inference-engine/thirdparty/ade/ade/source/memory_descriptor.cpp \
	inference-engine/thirdparty/ade/ade/source/memory_descriptor_ref.cpp \
	inference-engine/thirdparty/ade/ade/source/memory_descriptor_view.cpp \
	inference-engine/thirdparty/ade/ade/source/metadata.cpp \
	inference-engine/thirdparty/ade/ade/source/metatypes.cpp \
	inference-engine/thirdparty/ade/ade/source/node.cpp \
	inference-engine/thirdparty/ade/ade/source/search.cpp \
	inference-engine/thirdparty/ade/ade/source/subgraphs.cpp \
	inference-engine/thirdparty/ade/ade/source/topological_sort.cpp \
	inference-engine/thirdparty/ade/ade/source/passes/communications.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/communication \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/execution_engine \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/helpers \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/memory \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/metatypes \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/ade/passes \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/common/include



LOCAL_CFLAGS += -std=c++11  -Wall -Wno-unknown-pragmas -Wno-strict-overflow -fPIC -Wformat -Wformat-security -fstack-protector-all
LOCAL_CFLAGS += -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers  -fexceptions -frtti -Wno-error
LOCAL_CFLAGS += -DENABLE_VPU -DENABLE_MYRIAD -DNDEBUG -DAKS

LOCAL_SHARED_LIBRARIES :=
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_STATIC_LIBRARY)
