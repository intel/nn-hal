//
// INTEL CONFIDENTIAL
// Copyright 2017-2018 Intel Corporation.
//
// The source code contained or described herein and all documents
// related to the source code ("Material") are owned by Intel Corporation
// or its suppliers or licensors. Title to the Material remains with
// Intel Corporation or its suppliers and licensors. The Material may
// contain trade secrets and proprietary and confidential information
// of Intel Corporation and its suppliers and licensors, and is protected
// by worldwide copyright and trade secret laws and treaty provisions.
// No part of the Material may be used, copied, reproduced, modified,
// published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
//
// No license under any patent, copyright, trade secret or other
// intellectual property right is granted to or conferred upon you by
// disclosure or delivery of the Materials, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Include any supplier copyright notices as supplier requires Intel to use.
//
// Include supplier trademarks or logos as supplier requires Intel to use,
// preceded by an asterisk. An asterisked footnote can be added as follows:
// *Third Party trademarks are the property of their respective owners.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter
// this notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.
//

#pragma once

#include "mv_common.h"

namespace VPU {

const uint32_t EI_NIDENT = 2;  // 16?

PACKED(ElfN_Ehdr {
    uint8_t  e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint16_t e_version;  // uint32_t?
    uint32_t e_entry;
    uint32_t e_phoff;
    uint32_t e_shoff;
    uint16_t e_flags;  // uint32_t?
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};)

PACKED(mv_blob_header {
    uint32_t magic_number;
    uint32_t file_size;
    uint32_t blob_ver_major;
    uint32_t blob_ver_minor;
    uint32_t num_shaves;
    uint32_t bss_mem_size;
    uint32_t stage_section_offset;
    uint32_t buffer_section_offset;
    uint32_t relocation_section_offset;
};)

PACKED(mv_buffer_section_header {
    uint32_t buffer_section_size;
    uint8_t unused[12];
};)

PACKED(mv_relocation_section_header {
    uint32_t relocation_buffer_size;
    uint32_t blob_buffer_reloc_offset;
    uint32_t blob_buffer_reloc_size;
    uint32_t work_buffer_reloc_offset;
    uint32_t work_buffer_reloc_size;
};)

PACKED(mv_reloc_info {
    uint32_t offset;
    uint32_t location;
};)

PACKED(mv_stage_section_header {
    uint32_t stage_count;
    uint32_t stage_section_size;
    uint32_t input_size;
    uint32_t output_size;
};)

PACKED(mv_stage_header {
    uint32_t next_stage;
    uint32_t stage_type;
    uint32_t implementation_flag;
};)

enum cnnOperationMode {
    MODE_1_256 = 0,
    MODE_2_128 = 1,
    MODE_4_64  = 2,
    MODE_8_32  = 3,
    MODE_16_16 = 4,
    MODE_AUTO  = 5  // This is old behavior of automatic selection at runtime
};

enum cnnCoefficientMode {
    FP16_COEFF = 0,
    U8F_COEFF  = 1,
    FOUR_BIT_PLLTZD = 2,
    TWO_BIT_PLLTZD = 3,
    ONE_BIT_PLLTZD = 4,
    ONE_BIT_DIRECT = 5
};

enum cnnDataMode {
    MODE_FP16 = 0,
    MODE_U8F  = 1
};

enum cnnPadMode {
    PAD_WITH_ZEROS = 0x00,
    PAD_REPEAT_RIGHT_EDGE = 0x01,
    PAD_REPEAT_LEFT_EDGE = 0x08,
    PAD_REPEAT_TOP_EDGE = 0x04,
    PAD_REPEAT_BOTTOM_EDGE = 0x02
};

enum cnnPoolType {
    POOL_MAX = 0,
    POOL_AVERAGE = 1
};

enum cnnOperationType {
    TYPE_CONV = 0,
    TYPE_CONVPOOL = 1,
    TYPE_FULLCONN = 2,
    TYPE_POOL = 4
};

PACKED(cnnDescriptor {
    struct {
        uint32_t linkAddress : 32;
        cnnOperationType type : 3;
        cnnOperationMode mode : 3;
        uint32_t rsvd0 : 2;
        uint32_t id : 4;
        uint32_t it : 4;
        cnnCoefficientMode cm : 3;
        cnnDataMode dm : 1;
        uint32_t disInt : 1;
        uint32_t rsvd1 : 11;
    } Line0;

    union {
        struct {
            uint32_t inputHeight : 12;
            uint32_t rsvd0 : 4;
            uint32_t inputWidth : 12;
            uint32_t rsvd1 : 4;
            uint32_t inputChannels : 11;
            uint32_t rsvd2 : 5;
            uint32_t outputChannels : 11;
            uint32_t rsvd3 : 5;
        } ConvolutionPooling;

        struct {
            uint32_t inputWidth : 12;
            uint32_t rsvd0 : 20;
            uint32_t vectors : 8;
            uint32_t rsvd1 : 8;
            uint32_t vectors2 : 8;
            uint32_t rsvd2 : 8;
        } FullyConnected;
    } Line1;

    union {
        struct {
            uint32_t chPerRamBlock : 11;
            uint32_t rsvd0 : 5;
            uint32_t chStride : 4;
            uint32_t rsvd1 : 12;
            uint32_t kernelWidth : 4;
            uint32_t kernelHeight : 4;
            uint32_t rsvd2 : 19;
            cnnPadMode padType : 4;
            uint32_t padEn : 1;
        } ConvolutionPooling;

        struct {
            uint32_t dataPerRamBlock : 9;
            uint32_t rsvd0 : 23;
            uint32_t rsvd1 : 32;
        } FullyConnected;
    } Line2;

    union {
        struct {
            uint32_t poolEn : 1;
            uint32_t rsvd0 : 15;
            uint32_t poolKernelHeight : 8;
            uint32_t poolKernelWidth : 8;
            uint32_t avgPoolX : 16;
            uint32_t rsvd2 : 15;
            cnnPoolType poolType : 1;
        } ConvolutionPooling;

        struct {
            uint32_t rsvd0 : 1;
            uint32_t actualOutChannels : 8;  // Custom info (How many of the output channels contain useful info)
            uint32_t rsvd1 : 23;
            uint32_t X : 16;
            uint32_t rsvd2 : 16;
        } FullyConnected;
    } Line3;

    struct {
        uint32_t dataBaseAddr : 32;
        uint32_t t0 : 10;
        uint32_t a0 : 10;
        uint32_t a1 : 10;
        uint32_t reluxEn : 1;
        uint32_t reluEn : 1;
    } Line4;

    struct {
        uint32_t dataChStr : 32;
        uint32_t dataLnStr : 32;
    } Line5;

    union {
        struct {
            uint32_t coeffBaseAddr : 32;
            uint32_t coeffChStrOut : 32;
        } Convolution;

        struct {
            uint32_t vectorBaseAddr : 32;
            uint32_t vectorStrOut : 32;
        } FullyConnected;
    } Line6;

    union {
        struct {
            uint32_t coeffChStrIn : 32;
            uint32_t outLnStr : 32;
        } ConvolutionPooling;

        struct {
            uint32_t vectorStrIn : 32;
            uint32_t outLnStr : 32;
        } FullyConnected;
    } Line7;

    struct {
        uint32_t outBaseAddr : 32;
        uint32_t outChStr : 32;
    } Line8;

    union {
        struct {
            uint32_t localLs : 9;
            uint32_t rsvd0 : 7;
            uint32_t localCs : 13;
            uint32_t rsvd1 : 3;
            uint32_t linesPerCh : 9;
            uint32_t rsvd2 : 22;
            uint32_t rud : 1;
        } ConvolutionPooling;

        struct {
            uint32_t localLs : 9;
            uint32_t rsvd0 : 7;
            uint32_t localBs : 13;
            uint32_t rsvd1 : 3;
            uint32_t rsvd2 : 31;
            uint32_t rud : 1;
        } FullyConnected;
    } Line9;

    union {
        struct {
            uint32_t minLines : 9;
            uint32_t rsvd0 : 23;
            uint32_t coeffLpb : 8;
            uint32_t css : 8;
            uint32_t outputX : 12;
            uint32_t rsvd1 : 4;
        } ConvolutionPooling;

        struct {
            uint32_t rsvd0 : 16;
            uint32_t acc : 1;
            uint32_t rsvd1 : 15;
            uint32_t vectorLPB : 8;
            uint32_t rsvd2 : 8;
            uint32_t outputX : 12;  // Due to a hardware bug, outputX for FC must be set to 1
            uint32_t rsvd3 : 4;
        } FullyConnected;
    } Line10;

    struct {
        uint32_t biasBaseAddr : 32;
        uint32_t scaleBaseAddr : 32;
    } Line11;

    struct {
        uint32_t p0 : 16;
        uint32_t p1 : 16;
        uint32_t p2 : 16;
        uint32_t p3 : 16;
    } Line12;

    struct {
        uint32_t p4 : 16;
        uint32_t p5 : 16;
        uint32_t p6 : 16;
        uint32_t p7 : 16;
    } Line13;

    struct {
        uint32_t p8 : 16;
        uint32_t p9 : 16;
        uint32_t p10 : 16;
        uint32_t p11 : 16;
    } Line14;

    struct {
        uint32_t p12 : 16;
        uint32_t p13 : 16;
        uint32_t p14 : 16;
        uint32_t p15 : 16;
    } Line15;
};)

}  //  namespace VPU
