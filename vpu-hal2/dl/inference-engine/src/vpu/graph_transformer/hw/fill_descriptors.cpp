//
// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation.
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

#include "common.hpp"
#include <algorithm>

namespace {

// Configures the basic elements of a cnn descriptor
void cnnInitDescriptor(cnnDescriptor *desc,
                       cnnDataMode dataMode,
                       uint32_t Id, uint32_t disableInt, uint32_t interruptTrigger,
                       uint32_t linkAddress) {
    std::memset(desc, 0, sizeof(*desc));

    desc->Line0.disInt = disableInt & 0x01;
    desc->Line0.dm = dataMode;
    desc->Line0.it = interruptTrigger & 0x0F;
    desc->Line0.id = Id;
    desc->Line0.linkAddress = linkAddress;
}

// Configures the input parameters of a cnn descriptor (only for convolution or pooling layers)
void cnnSetupInput(cnnDescriptor *desc,
                   uint32_t inputData,
                   uint32_t noOfChannels,
                   uint32_t chStride, uint32_t lineStride,
                   uint32_t dataWidth, uint32_t dataHeight) {
    desc->Line1.ConvolutionPooling.inputHeight = dataHeight - 1u;
    desc->Line1.ConvolutionPooling.inputWidth = dataWidth - 1u;
    desc->Line1.ConvolutionPooling.inputChannels = noOfChannels - 1u;
    desc->Line4.dataBaseAddr = inputData;
    desc->Line5.dataChStr = chStride;
    desc->Line5.dataLnStr = lineStride;
}

// Configures the output parameters of a cnn descriptor (only for convolution or pooling layers)
void cnnSetupOutput(cnnDescriptor *desc,
                    uint32_t outputData,
                    uint32_t noOfChannels,
                    uint32_t chStride, uint32_t lineStride) {
    desc->Line1.ConvolutionPooling.outputChannels = noOfChannels - 1u;
    desc->Line7.ConvolutionPooling.outLnStr = lineStride;
    desc->Line8.outChStr = chStride;
    desc->Line8.outBaseAddr = outputData;
}

// Configures the coefficients parameters of a cnn descriptor (only for convolution layer)
void cnnSetupConvolutionCoefficients(cnnDescriptor *desc,
                                     cnnCoefficientMode mode,
                                     uint32_t coefficientData,
                                     uint32_t kernelWidth, uint32_t kernelHeight,
                                     uint32_t convolutionStride,
                                     float *palette = nullptr) {
    desc->Line6.Convolution.coeffBaseAddr = coefficientData;
    desc->Line2.ConvolutionPooling.kernelHeight = kernelHeight - 1u;
    desc->Line2.ConvolutionPooling.kernelWidth = kernelWidth - 1u;
    desc->Line2.ConvolutionPooling.chStride = convolutionStride - 1u;
    desc->Line0.cm = mode;
    desc->Line0.type = TYPE_CONV;

    if (palette != nullptr) {
        desc->Line12.p0 = PrecisionUtils::f32tof16(palette[0]);
        desc->Line12.p1 = PrecisionUtils::f32tof16(palette[1]);
        desc->Line12.p2 = PrecisionUtils::f32tof16(palette[2]);
        desc->Line12.p3 = PrecisionUtils::f32tof16(palette[3]);
        desc->Line13.p4 = PrecisionUtils::f32tof16(palette[4]);
        desc->Line13.p5 = PrecisionUtils::f32tof16(palette[5]);
        desc->Line13.p6 = PrecisionUtils::f32tof16(palette[6]);
        desc->Line13.p7 = PrecisionUtils::f32tof16(palette[7]);
        desc->Line14.p8 = PrecisionUtils::f32tof16(palette[8]);
        desc->Line14.p9 = PrecisionUtils::f32tof16(palette[9]);
        desc->Line14.p10 = PrecisionUtils::f32tof16(palette[10]);
        desc->Line14.p11 = PrecisionUtils::f32tof16(palette[11]);
        desc->Line15.p12 = PrecisionUtils::f32tof16(palette[12]);
        desc->Line15.p13 = PrecisionUtils::f32tof16(palette[13]);
        desc->Line15.p14 = PrecisionUtils::f32tof16(palette[14]);
        desc->Line15.p15 = PrecisionUtils::f32tof16(palette[15]);
    }
}

// Configures the optional pooling parameters for a cnn descriptor (only for convolution layers)
void cnnSetupConvolutionPooling(cnnDescriptor *desc, uint32_t kernelWidth, uint32_t kernelHeight) {
    desc->Line0.type = TYPE_CONVPOOL;
    desc->Line3.ConvolutionPooling.poolEn = 1u;
    desc->Line3.ConvolutionPooling.poolKernelHeight = kernelHeight - 1u;
    desc->Line3.ConvolutionPooling.poolKernelWidth = kernelWidth - 1u;
    desc->Line3.ConvolutionPooling.poolType = POOL_MAX;
}

// Configures the pooling parameters for a cnn descriptor (only for a pooling layer)
void cnnSetupPoolingLayer(cnnDescriptor *desc,
                          cnnPoolType type,
                          uint32_t kernelWidth, uint32_t kernelHeight,
                          uint32_t poolStride) {
    desc->Line0.type = TYPE_POOL;
    desc->Line3.ConvolutionPooling.poolEn = 1u;
    desc->Line3.ConvolutionPooling.poolKernelHeight = kernelHeight - 1u;
    desc->Line3.ConvolutionPooling.poolKernelWidth = kernelWidth - 1u;
    desc->Line2.ConvolutionPooling.chStride = poolStride - 1u;
    desc->Line3.ConvolutionPooling.poolType = type;
    if (type == POOL_AVERAGE) {
        desc->Line3.ConvolutionPooling.avgPoolX = PrecisionUtils::f32tof16(1.0f / (kernelWidth * kernelHeight));
    }
}

// Configures the input parameters for a cnn descriptor (only for a fully connected layer)
void cnnSetupInputFullyConnectedLayer(cnnDescriptor *desc,
                                      uint32_t inputData,
                                      uint8_t accumulate,
                                      uint32_t inputWidth) {
    desc->Line0.type = TYPE_FULLCONN;
    desc->Line1.FullyConnected.inputWidth = inputWidth - 1u;
    desc->Line4.dataBaseAddr = inputData;
    desc->Line10.FullyConnected.acc = accumulate;
}

// Configures the output parameters for a cnn descriptor (only for a fully connected layer)
void cnnSetupOutputFullyConnectedLayer(cnnDescriptor *desc, uint32_t outputData) {
    desc->Line0.type = TYPE_FULLCONN;
    desc->Line8.outBaseAddr = outputData;
}

// Configures the vector parameters for a cnn descriptor (only for a fully connected layer)
void cnnSetupVectorsFullyConnectedLayer(cnnDescriptor *desc,
                                        cnnCoefficientMode mode,
                                        uint32_t vectorData,
                                        uint32_t noOfVectors,
                                        float *palette = nullptr) {
    desc->Line0.type = TYPE_FULLCONN;
    desc->Line0.cm = mode;
    desc->Line1.FullyConnected.vectors = noOfVectors - 1u;
    desc->Line1.FullyConnected.vectors2 = noOfVectors - 1u;
    desc->Line6.FullyConnected.vectorBaseAddr = vectorData;

    if (palette != nullptr) {
        desc->Line12.p0 = PrecisionUtils::f32tof16(palette[0]);
        desc->Line12.p1 = PrecisionUtils::f32tof16(palette[1]);
        desc->Line12.p2 = PrecisionUtils::f32tof16(palette[2]);
        desc->Line12.p3 = PrecisionUtils::f32tof16(palette[3]);
        desc->Line13.p4 = PrecisionUtils::f32tof16(palette[4]);
        desc->Line13.p5 = PrecisionUtils::f32tof16(palette[5]);
        desc->Line13.p6 = PrecisionUtils::f32tof16(palette[6]);
        desc->Line13.p7 = PrecisionUtils::f32tof16(palette[7]);
        desc->Line14.p8 = PrecisionUtils::f32tof16(palette[8]);
        desc->Line14.p9 = PrecisionUtils::f32tof16(palette[9]);
        desc->Line14.p10 = PrecisionUtils::f32tof16(palette[10]);
        desc->Line14.p11 = PrecisionUtils::f32tof16(palette[11]);
        desc->Line15.p12 = PrecisionUtils::f32tof16(palette[12]);
        desc->Line15.p13 = PrecisionUtils::f32tof16(palette[13]);
        desc->Line15.p14 = PrecisionUtils::f32tof16(palette[14]);
        desc->Line15.p15 = PrecisionUtils::f32tof16(palette[15]);
    }
}

// Configures the optional padding parameters for a cnn descriptor
void cnnSetupPadding(cnnDescriptor *desc, uint32_t mode) {
    desc->Line2.ConvolutionPooling.padEn = 1u;
    desc->Line2.ConvolutionPooling.padType = static_cast<cnnPadMode>(mode);
}

// Configures the optional ReLU parameters for a cnn descriptor (only for convolution and fully connected layers)
void cnnSetupReLU(cnnDescriptor *desc, uint16_t t0, uint16_t a0, uint16_t a1) {
    desc->Line4.reluEn = 1u;
    desc->Line4.t0 = t0;
    desc->Line4.a0 = a0;
    desc->Line4.a1 = a1;
}

// Configures the optional ReLUX parameters for a cnn descriptor (only for convolution and fully connected layers)
void cnnSetupReLUX(cnnDescriptor *desc, uint16_t x) {
    desc->Line4.reluxEn = 1u;
    desc->Line3.ConvolutionPooling.avgPoolX = x;
}

// Configures the optional bias parameter for a cnn descriptor (only for convolution and fully connected layers)
void cnnSetupBias(cnnDescriptor *desc, uint32_t biasData) {
    desc->Line11.biasBaseAddr = biasData;
}

// Configures the optional scale parameter for a cnn descriptor (only for convolution and fully connected layers)
void cnnSetupScale(cnnDescriptor *desc, uint32_t scaleData) {
    desc->Line11.scaleBaseAddr = scaleData;
}

// Configures the layer to reuse the data from the previous operation which is already stored in the Data RAM (max 128kb)
void cnnReuseData(cnnDescriptor *desc) {
    desc->Line9.ConvolutionPooling.rud = 1;
}

const uint32_t COEFF_LINE_SIZE = 8 * sizeof(uint16_t);
const uint32_t LOCAL_RAM_SIZE = 128 * 1024;
const uint32_t CMX_DATA_BIT_WIDTH = 128;

const uint32_t COEFF_PER_WORD_VALUES[] = {1, 2, 4, 8, 16, 16};

// Configure strides and dependent variables for a convolution layer and checks for validity
void cnnFinalizeConvolutionLayer(cnnDescriptor *desc, cnnOperationMode mode, uint32_t outputX) {
    uint32_t noOfBlocks = 1u << mode;
    uint32_t sizeOfBlock = LOCAL_RAM_SIZE >> mode;
    uint32_t bytesPerPixel = 1u << (1u - desc->Line0.dm);
    uint32_t pixelsPerCMXLine = CMX_DATA_BIT_WIDTH / (bytesPerPixel * 8u);
    uint32_t coeffPerWord = COEFF_PER_WORD_VALUES[desc->Line0.cm];
    uint32_t inputChannels = desc->Line1.ConvolutionPooling.inputChannels + 1u;
    uint32_t inputWidth = desc->Line1.ConvolutionPooling.inputWidth + 1u;
    uint32_t inputHeight = desc->Line1.ConvolutionPooling.inputHeight + 1u;
    uint32_t kernelHeight = desc->Line2.ConvolutionPooling.kernelHeight + 1u;
    uint32_t kernelWidth = desc->Line2.ConvolutionPooling.kernelWidth + 1u;
    uint32_t poolKernelHeight = desc->Line3.ConvolutionPooling.poolKernelHeight + 1u;

    assert(inputChannels % noOfBlocks == 0);

    // Calculate local line stride
    uint32_t localLineStride = (inputWidth + (pixelsPerCMXLine - 1u)) / pixelsPerCMXLine;

    uint32_t chanPerBlock = inputChannels / noOfBlocks;
    uint32_t availableBytesPerChan = sizeOfBlock / chanPerBlock;
    uint32_t bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel;

    // Calculate lines per channel
    uint32_t linesPerChan = availableBytesPerChan / bytesPerLine;
    if (linesPerChan > inputHeight) {
        linesPerChan = inputHeight;
    }

    uint32_t localChanStride = linesPerChan * localLineStride;

    uint32_t minLines = 0u;
    if (desc->Line3.ConvolutionPooling.poolEn == 1u) {
        minLines = kernelHeight + poolKernelHeight;  // or is it max between the two ???
    } else {
        minLines = std::min(kernelHeight + 1, linesPerChan);
    }
    assert(minLines <= linesPerChan);

    uint32_t coeffLPB = (chanPerBlock * kernelHeight * kernelWidth + coeffPerWord - 1) / coeffPerWord;
    assert(coeffLPB <= 256);

    uint32_t coeffSetSize = kernelHeight * kernelWidth;

    // Calculate coefficient strides
    uint32_t coeffChStrideIn = desc->Line7.ConvolutionPooling.coeffChStrIn;
    uint32_t coeffChStrideOut = desc->Line6.Convolution.coeffChStrOut;
    uint32_t bytesPerCoeffSet = coeffSetSize;
    if (coeffChStrideIn == 0) {
        // Coefficient strides were not given by the user -> compute them
        switch (desc->Line0.cm) {
        case FP16_COEFF:
            // Coeff stride in is the distance between 2 input channels
            bytesPerCoeffSet *= sizeof(ie_fp16);
            coeffChStrideIn = bytesPerCoeffSet * 8u;
            coeffChStrideOut = coeffChStrideIn * inputChannels;
            break;
        case U8F_COEFF:
        case FOUR_BIT_PLLTZD:
        case TWO_BIT_PLLTZD:
        case ONE_BIT_PLLTZD:
        case ONE_BIT_DIRECT:
            bytesPerCoeffSet *= chanPerBlock;
            bytesPerCoeffSet = (bytesPerCoeffSet + coeffPerWord - 1u) / coeffPerWord;
            bytesPerCoeffSet *= sizeof(ie_fp16) * 8u;
            coeffChStrideIn = bytesPerCoeffSet / chanPerBlock;
            coeffChStrideOut = coeffChStrideIn * chanPerBlock * noOfBlocks;
            break;
        }
    }

    // Update cnn structure with computed data
    desc->Line0.mode = mode;
    desc->Line2.ConvolutionPooling.chPerRamBlock = chanPerBlock - 1u;
    desc->Line7.ConvolutionPooling.coeffChStrIn = coeffChStrideIn;
    desc->Line6.Convolution.coeffChStrOut = coeffChStrideOut;
    desc->Line9.ConvolutionPooling.linesPerCh = linesPerChan - 1u;
    desc->Line9.ConvolutionPooling.localLs = localLineStride;
    desc->Line9.ConvolutionPooling.localCs = localChanStride;
    desc->Line10.ConvolutionPooling.minLines = minLines - 1u;
    desc->Line10.ConvolutionPooling.coeffLpb = coeffLPB - 1u;
    desc->Line10.ConvolutionPooling.css = coeffSetSize - 1u;
    desc->Line10.ConvolutionPooling.outputX = outputX;
}

// Configure strides and dependent variables for a pooling layer and checks for validity
void cnnFinalizePoolingLayer(cnnDescriptor *desc, cnnOperationMode mode, uint32_t outputX) {
    uint32_t sizeOfBlock = LOCAL_RAM_SIZE;
    uint32_t bytesPerPixel = 1u << (1u - desc->Line0.dm);
    uint32_t pixelsPerCMXLine = CMX_DATA_BIT_WIDTH / (bytesPerPixel * 8u);

    uint32_t inputWidth = desc->Line1.ConvolutionPooling.inputWidth + 1u;
    uint32_t inputHeight = desc->Line1.ConvolutionPooling.inputHeight + 1u;

    uint32_t poolKernelHeight = desc->Line3.ConvolutionPooling.poolKernelHeight + 1u;

    // Calculate local line stride
    uint32_t localLineStride = (inputWidth + (pixelsPerCMXLine - 1)) / pixelsPerCMXLine;

    // Get the optimal operation mode
    sizeOfBlock >>= mode;

    uint32_t chanPerBlock = 1u;
    uint32_t availableBytesPerChan = sizeOfBlock / chanPerBlock;
    uint32_t bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel;

    // Calculate lines per channel
    uint32_t linesPerChan = availableBytesPerChan / bytesPerLine;
    if (linesPerChan > inputHeight) {
        linesPerChan = inputHeight;
    }

    uint32_t localChanStride = linesPerChan * localLineStride;
    uint32_t minLines = poolKernelHeight;

    // Update cnn structure with computed data
    desc->Line0.mode = mode;
    desc->Line2.ConvolutionPooling.chPerRamBlock = chanPerBlock - 1u;
    desc->Line9.ConvolutionPooling.linesPerCh = linesPerChan - 1u;
    desc->Line9.ConvolutionPooling.localLs = localLineStride;
    desc->Line9.ConvolutionPooling.localCs = localChanStride;
    desc->Line10.ConvolutionPooling.minLines = minLines;
    desc->Line10.ConvolutionPooling.outputX = outputX;
}

// Configure strides and dependent variables for a fully connected layer and checks for validity
void cnnFinalizeFullyConnectedLayer(cnnDescriptor *desc,
                                    cnnOperationMode mode,
                                    uint32_t totalDimX,
                                    uint32_t descriptorOutputChannels) {
    uint32_t inputWidth = desc->Line1.FullyConnected.inputWidth + 1u;

    uint32_t noOfBlocks = (1u << mode);
    uint32_t pixelsPerBlock = inputWidth / noOfBlocks;

    // Calculate input strides
    uint32_t inDataLineStride = 16u;
    uint32_t inDataBlockStride = inDataLineStride * pixelsPerBlock;

    // Calculate output strides
    uint32_t outDataLineStride = 16u;
    uint32_t outDataBlockStride = outDataLineStride * pixelsPerBlock;

    // Calculate vector strides
    uint32_t vectStrideIn = pixelsPerBlock * 8u * sizeof(ie_fp16);
    uint32_t vectStrideOut = totalDimX * 8u * sizeof(ie_fp16);

    // Calculate local strides
    uint32_t localLineStride = 16u;
    uint32_t localBlkStr = localLineStride * pixelsPerBlock;

    uint32_t vectLPB = pixelsPerBlock;
    assert(vectLPB <= 256);

    // Update cnn structure with computed data
    desc->Line0.mode = mode;
    desc->Line2.FullyConnected.dataPerRamBlock = pixelsPerBlock - 1u;
    desc->Line9.FullyConnected.localLs = localLineStride;
    desc->Line9.FullyConnected.localBs = localBlkStr;
    desc->Line10.FullyConnected.vectorLPB = vectLPB - 1u;
    desc->Line5.dataLnStr = inDataLineStride;
    desc->Line5.dataChStr = inDataBlockStride;
    desc->Line7.FullyConnected.outLnStr = outDataLineStride;
    desc->Line8.outChStr = outDataBlockStride;
    desc->Line6.FullyConnected.vectorStrOut = vectStrideOut;
    desc->Line7.FullyConnected.vectorStrIn = vectStrideIn;
    desc->Line3.FullyConnected.actualOutChannels = descriptorOutputChannels - 1u;
}

}  // namespace

void GraphTransformerImpl::fillHWDescriptors() {
    for (auto& stage : _stages) {
        assert(stage != nullptr);

        if (stage->optimized)
            continue;

        if (stage->type == kMyriadXHwConvolution) {
            auto hwStage = std::dynamic_pointer_cast<VpuMyriadXHwConvolutionStage>(stage);
            assert(hwStage != nullptr);

            auto input = stage->inputs[0];
            auto biases = stage->inputs[2];
            auto output = stage->outputs[0];

            hwStage->descriptors.clear();

            uint32_t totalOutChans = 0;
            for (size_t i = 0; i < hwStage->tiles.size(); ++i) {
                totalOutChans += std::get<0>(hwStage->tiles[i]);
            }

            uint32_t outChanOffset = 0u;
            for (size_t outTileIndex = 0; outTileIndex < hwStage->tiles.size(); ++outTileIndex) {
                uint32_t outChans;
                cnnOperationMode mode;
                std::tie(outChans, mode) = hwStage->tiles[outTileIndex];

                cnnDescriptor desc = {};
                cnnInitDescriptor(
                            &desc,
                            MODE_FP16,
                            0u, 0u, 0u, 0u);
                cnnSetupInput(
                            &desc,
                            0u,
                            hwStage->newInputDimZ,
                            input->strides[Dim::Z], input->strides[Dim::Y],
                            input->dims[Dim::X], input->dims[Dim::Y]);
                cnnSetupOutput(
                            &desc,
                            outChanOffset * output->strides[Dim::Z],
                            outChans,
                            output->strides[Dim::Z], output->strides[Dim::Y]);
                cnnSetupConvolutionCoefficients(
                            &desc,
                            FP16_COEFF,
                            outChanOffset * hwStage->newInputDimZ * hwStage->radixX * hwStage->radixY * sizeof(ie_fp16),
                            hwStage->radixX, hwStage->radixY,
                            hwStage->stride);
                if (hwStage->withPool) {
                    cnnSetupConvolutionPooling(&desc, hwStage->poolRadX, hwStage->poolRadY);
                }
                if (hwStage->pad.enable) {
                    cnnSetupPadding(&desc, PAD_WITH_ZEROS);
                }
                if (hwStage->hasRelu) {
                    cnnSetupReLU(&desc, 0u, 0u, 1u);
                }
                if (biases->index != IndexNone) {
                    cnnSetupBias(&desc, outChanOffset * sizeof(ie_fp16));
                }
                cnnFinalizeConvolutionLayer(&desc, mode, output->dims[Dim::X]);

                hwStage->descriptors.push_back(desc);

                outChanOffset += outChans;
            }
        } else if (stage->type == kMyriadXHwFCL) {
            auto hwStage = std::dynamic_pointer_cast<VpuMyriadXHwFullyConnectedStage>(stage);
            assert(hwStage != nullptr);

            auto input = stage->inputs[0];
            auto biases = stage->inputs[2];
            auto output = stage->outputs[0];

            hwStage->descriptors.clear();

            uint32_t outputOffset = 0u;

            for (size_t tileIndex = 0; tileIndex < hwStage->tiles.size(); ++tileIndex) {
                const auto& subTiles = hwStage->tiles[tileIndex];

                auto lastTile = (tileIndex == hwStage->tiles.size() - 1);

                uint32_t inputOffset = 0u;
                uint32_t workOutN = 0;

                // Detect how many of the output channels the current subTiles are generating is actual/real
                // This is needed for the early interrupts workaround

                uint32_t outChansSoFar = 0;
                for (size_t i = 0; i < tileIndex + 1; ++i) {
                    outChansSoFar += std::get<1>(hwStage->tiles[i][0]);
                }

                uint32_t actualOutputChannels = output->dims[Dim::Z];

                uint32_t descriptorOutputChannels = 0u;
                if (outChansSoFar > actualOutputChannels) {
                    if (tileIndex == 0) {
                        descriptorOutputChannels = actualOutputChannels;
                    } else {
                        uint32_t runningOutputChans = 0u;
                        for (size_t i = 0; i < tileIndex; ++i) {
                            runningOutputChans += std::get<1>(hwStage->tiles[i][0]);
                        }

                        descriptorOutputChannels = actualOutputChannels - runningOutputChans;
                    }
                } else {
                    descriptorOutputChannels = std::get<1>(subTiles[0]);
                }

                for (size_t subTileIndex = 0; subTileIndex < subTiles.size(); ++subTileIndex) {
                    uint32_t workInN;
                    cnnOperationMode mode;
                    std::tie(workInN, workOutN, mode) = subTiles[subTileIndex];

                    bool lastSubTile = (subTileIndex == subTiles.size() - 1);

                    bool accum = !lastSubTile;

                    uint32_t tapsOffset = (outputOffset * hwStage->newInputDimZ) + (inputOffset * 8);

                    cnnDescriptor desc = {};
                    cnnInitDescriptor(
                                &desc,
                                MODE_FP16,
                                0u, 0u, 0u, 0u);
                    cnnSetupInputFullyConnectedLayer(
                                &desc,
                                inputOffset * input->strides[Dim::Z],
                                accum,
                                workInN);
                    cnnSetupOutputFullyConnectedLayer(
                                &desc,
                                outputOffset * output->strides[Dim::Z]);
                    cnnSetupVectorsFullyConnectedLayer(
                                &desc,
                                FP16_COEFF,
                                tapsOffset * sizeof(ie_fp16),
                                workOutN);
                    if (lastSubTile && hwStage->hasRelu) {
                        cnnSetupReLU(&desc, 0u, 0u, 1u);
                    }
                    if (biases->index != IndexNone) {
                        cnnSetupBias(&desc, outputOffset * sizeof(ie_fp16));
                    }
                    cnnFinalizeFullyConnectedLayer(&desc, mode, hwStage->newInputDimZ, descriptorOutputChannels);

                    hwStage->descriptors.push_back(desc);

                    inputOffset += workInN;
                }

                outputOffset += workOutN;
            }
        } else if (stage->type == kMyriadXHwPooling) {
            auto hwStage = std::dynamic_pointer_cast<VpuMyriadXHwPoolingStage>(stage);
            assert(hwStage != nullptr);

            auto input = stage->inputs[0];
            auto output = stage->outputs[0];

            hwStage->descriptors.clear();

            uint32_t totalOutChans = 0;
            for (const auto& t : hwStage->tiles) {
                totalOutChans += std::get<0>(t);
            }

            uint32_t chanOffset = 0;

            for (size_t tileIndex = 0; tileIndex < hwStage->tiles.size(); ++tileIndex) {
                uint32_t outChans;
                cnnOperationMode mode;
                std::tie(outChans, mode) = hwStage->tiles[tileIndex];

                cnnDescriptor desc = {};
                cnnInitDescriptor(
                            &desc,
                            MODE_FP16,
                            0u, 0u, 0u, 0u);
                cnnSetupInput(
                            &desc,
                            chanOffset * input->strides[Dim::Z],
                            outChans,
                            input->strides[Dim::Z], input->strides[Dim::Y],
                            input->dims[Dim::X], input->dims[Dim::Y]);
                cnnSetupOutput(
                            &desc,
                            chanOffset * output->strides[Dim::Z],
                            outChans,
                            output->strides[Dim::Z], output->strides[Dim::Y]);
                cnnSetupPoolingLayer(
                            &desc,
                            hwStage->poolType,
                            hwStage->radixX, hwStage->radixY,
                            hwStage->stride);
                if (hwStage->pad.enable) {
                    uint32_t padType = 0;

                    if (hwStage->poolType == POOL_MAX) {
                        // Repeat padding is not used in CNNs at all
                        if (hwStage->pad.left > 0)
                            padType |= PAD_REPEAT_LEFT_EDGE;
                        if (hwStage->pad.right > 0)
                            padType |= PAD_REPEAT_RIGHT_EDGE;
                        if (hwStage->pad.top > 0)
                            padType |= PAD_REPEAT_TOP_EDGE;
                        if (hwStage->pad.bottom > 0)
                            padType |= PAD_REPEAT_BOTTOM_EDGE;
                    }

                    cnnSetupPadding(&desc, padType);
                }
                if (hwStage->hasRelu) {
                    cnnSetupReLU(&desc, 0u, 0u, 1u);
                }
                cnnFinalizePoolingLayer(&desc, mode, output->dims[Dim::X]);

                hwStage->descriptors.push_back(desc);

                chanOffset += outChans;
            }
        }
    }
}
