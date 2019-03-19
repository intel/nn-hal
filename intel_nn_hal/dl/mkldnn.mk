LOCAL_PATH := $(call my-dir)/../../../dldt

include $(CLEAR_VARS)

LOCAL_MODULE := libmkldnn
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/common \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu/gemm \
	$(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu/xbyak

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/src/mkldnn_plugin \
	$(LOCAL_PATH)/inference-engine/src/mkldnn_plugin/mkldnn \
	$(LOCAL_PATH)/inference-engine/src/mkldnn_plugin/nodes

LOCAL_CFLAGS += \
			-std=c++11 \
			-fvisibility=internal \
			-fvisibility-inlines-hidden \
			-Wall \
			-Werror \
			-Wno-pass-failed \
			-Wno-error=unused-private-field \
			-Wno-unknown-pragmas \
			-Wno-strict-overflow \
			-fPIC \
			-fPIE \
			-Wformat \
			-Wformat-security \
			-fstack-protector-all \
			-O3 \
			-DNDEBUG \
			-D_FORTIFY_SOURCE=2 \
			-Wno-unused-variable \
			-Wno-unused-parameter \
			-Wno-non-virtual-dtor \
			-Wno-missing-field-initializers \
			-Wno-shift-count-overflow \
			-fexceptions \
			-DMKLDNN_DLL \
			-DMKLDNN_DLL_EXPORTS \
			-D__STDC_CONSTANT_MACROS \
			-D__STDC_LIMIT_MACROS \
			-Dmkldnn_EXPORTS \
			-fopenmp \
			-frtti
LOCAL_STATIC_LIBRARIES := libomp

LOCAL_SRC_FILES += \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_i8i8_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_softmax.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_memory.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_reducer.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_roi_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_transpose_src_utils.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_sse42_1x1_conv_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_eltwise.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_sse42_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_x8s8s32x_conv_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_barrier.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm_x8s8s32x_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_fp32_wino_conv_4x3.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_concat.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder_utils.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_u8s8s32x_deconvolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_depthwise.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_reorder.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_common_conv_winograd_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_dw_conv_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_batch_normalization.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_x8s8s32x_dw_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_x8s8s32x_1x1_conv_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_rnn.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_deconvolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm_convolution_utils.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_shuffle.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_u8s8s32x_wino_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_i8i8_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/simple_sum.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_x8s8s32x_1x1_conv_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_common_lrn.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_eltwise.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_fp32_wino_conv_2x3.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_dw_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/simple_concat.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_depthwise.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_common_1x1_conv_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_x8s8s32x_1x1_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_lrn_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_softmax.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm_inner_product.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/nspc_batch_normalization.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/nchw_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_sse42_conv_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_engine.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_softmax_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_x8s8s32x_conv_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_batch_normalization_utils.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_x8s8s32x_1x1_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm_u8s8s32x_inner_product.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_pool_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_lrn.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_common_conv_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_x8s8s32x_dw_conv_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_common_convolution_winograd.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_common_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_common_1x1_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_x8s8s32x_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_roi_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_batch_normalization.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_x8s8s32x_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_lrn.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_sum.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_roi_pool_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx512_core_fp32_wino_conv_4x3_kernel.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ncsp_batch_normalization.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/nhwc_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_conv_kernel_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_sse42_1x1_convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/ref_inner_product.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/jit_avx_gemm_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/jit_avx512_common_gemm_f32.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/gemm.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/gemm_utils.cpp \
inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/ref_gemm.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/deconvolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/rnn.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/shuffle.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/primitive_desc.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/query.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/reorder.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/verbose.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/convolution_relu.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/stream.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/softmax.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/primitive.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/convolution.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/inner_product.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/eltwise.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/engine.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/lrn.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/roi_pooling.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/primitive_attr.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/memory_desc_wrapper.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/mkldnn_debug.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/depthwise.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp \
inference-engine/thirdparty/mkl-dnn/src/common/batch_normalization.cpp


include $(BUILD_SHARED_LIBRARY)
