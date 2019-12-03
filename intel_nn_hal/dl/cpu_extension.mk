LOCAL_PATH := $(call my-dir)/../../../dldt

include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_extension
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/src/extension \
	$(LOCAL_PATH)/inference-engine/src/extension/common \

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
	-D_FORTIFY_SOURCE=2 \
	-DCI_BUILD_NUMBER='""'

LOCAL_CFLAGS += \
	-DNDEBUG \
	-D__ANDROID__ -DNNLOG -DDEBUG \
	-DIMPLEMENT_INFERENCE_ENGINE_API \
	-DIMPLEMENT_INFERENCE_ENGINE_PLUGIN \
	-DIE_THREAD=IE_THREAD_OMP \

LOCAL_STATIC_LIBRARIES := libomp
LOCAL_SHARED_LIBRARIES := libinference_engine


LOCAL_SRC_FILES += \
	inference-engine/src/extension/ext_proposal_onnx.cpp \
	inference-engine/src/extension/ext_expand.cpp \
	inference-engine/src/extension/ext_mvn.cpp \
	inference-engine/src/extension/ext_psroi.cpp \
	inference-engine/src/extension/ext_simplernms.cpp \
	inference-engine/src/extension/ext_list.cpp \
	inference-engine/src/extension/ext_interp.cpp \
	inference-engine/src/extension/ext_base.cpp \
	inference-engine/src/extension/ext_pad.cpp \
	inference-engine/src/extension/ext_space_to_depth.cpp \
	inference-engine/src/extension/ext_proposal.cpp \
	inference-engine/src/extension/ext_powerfile.cpp \
	inference-engine/src/extension/ext_argmax.cpp \
	inference-engine/src/extension/ext_grn.cpp \
	inference-engine/src/extension/ext_detectionoutput_onnx.cpp \
	inference-engine/src/extension/ext_roifeatureextractor_onnx.cpp \
	inference-engine/src/extension/simple_copy.cpp \
	inference-engine/src/extension/ext_gather.cpp \
	inference-engine/src/extension/ext_topkrois_onnx.cpp \
	inference-engine/src/extension/ext_priorbox.cpp \
	inference-engine/src/extension/ext_priorbox_clustered.cpp \
	inference-engine/src/extension/ext_ctc_greedy.cpp \
	inference-engine/src/extension/ext_strided_slice.cpp \
	inference-engine/src/extension/ext_unsqueeze.cpp \
	inference-engine/src/extension/ext_depth_to_space.cpp \
	inference-engine/src/extension/ext_priorgridgenerator_onnx.cpp \
	inference-engine/src/extension/ext_shuffle_channels.cpp \
	inference-engine/src/extension/ext_reorg_yolo.cpp \
	inference-engine/src/extension/ext_resample.cpp \
	inference-engine/src/extension/ext_fill.cpp \
	inference-engine/src/extension/ext_detectionoutput.cpp \
	inference-engine/src/extension/ext_region_yolo.cpp \
	inference-engine/src/extension/ext_reverse_sequence.cpp \
	inference-engine/src/extension/ext_range.cpp \
	inference-engine/src/extension/ext_normalize.cpp \
	inference-engine/src/extension/ext_squeeze.cpp \

include $(BUILD_SHARED_LIBRARY)
