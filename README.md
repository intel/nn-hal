![CI](https://github.com/reaganlo/nn-hal/actions/workflows/ci.yml/badge.svg)

# ChromeOS Neural Networks HAL with OpenVINO supporting hardware accelerators such as /
Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)
Intel® Movidius™ Vision Processing Units (VPUs)

## Introduction
The ChromeOS Neural Network Hardware Abstraction Layer(NN HAL) provides the hardware acceleration
for ChromeOS Neural Networks (NN) API. Intel NN-HAL takes the advantage of the Intel OpenVINO
CPUPlugin, based on MKLDNN which enables high performance and low power implementation of Neural Networks API.
Intel OpenVINO is available at https://github.com/openvinotoolkit/openvino

## OpenVINO version
This version of the HAL works with OpenVINO 2022.1.1 branch: https://github.com/openvinotoolkit/openvino/tree/releases/2022/1.1

## Supported Operations
Following operations are currently supported by Android Neural Networks HAL for Intel MKL-DNN.

ADD
AVERAGE_POOL_2D
CONCATENATION
CONV_2D
Convolution2DTransposeBias*
DEPTHWISE_CONV_2D
DEQUANTIZE
FULLY_CONNECTED
LOGISTIC
MAX_POOL_2D
MUL
PAD
RELU
RELU6
RESHAPE
RESIZE_BILINEAR

## License
ChromeOS Neural Networks HAL is distributed under the Apache License, Version 2.0
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0


## How to provide feedback
By default, please submit an issue using native github.com interface:
https://github.com/intel/nn-hal/issues

## How to contribute

Create a pull request on github.com with your patch. Make sure your change is cleanly building
and passing ULTs.

A maintainer will contact you if there are questions or concerns.

## Continuous Integration
Before committing any changes, make sure the coding style and testing configs are correct.
If not, the CI will fail.

### Coding Style

Run the following command to ensure that the proper coding style is being followed:
```
    find . -regex '.*\.\(cpp\|hpp\|cc\|cxx\|h\)' -exec clang-format -style=file -i {} \;
```

### Build and Test

Update the BOARD value in [build-test.sh](ci/build-test.sh) as per your test requirement.
If your BOARD is not supported, please contact the maintainer to get it added.

Currently, the CI builds the intel-nnhal package and runs the following tests:
- Functional tests that include ml_cmdline and a subset of cts and vts tests.
