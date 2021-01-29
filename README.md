# Android Neural Networks HAL with OpenVINO supporting hardware accelerators such as /
Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)

## Introduction
The Android Neural Network Hardware Abstraction Layer(NN HAL) provides the hardware accelration
for Android Neural Networks (NN) API. Intel NN-HAL takes the advantage of the Intel MKLD-DNN,
enables high performance and low power implementation of Neural Networks API.
Intel MKL-DNN https://github.com/intel/mkl-dnn &  https://01.org/mkl-dnn
Android NN API is on [Neural Networks API]
(https://developer.android.com/ndk/guides/neuralnetworks/index.html).
OpenVINO deep learning framework https://github.com/opencv/dldt & https://01.org/openvinotoolkit


## Supported Operations
Following operations are currently supported by Android Neural Networks HAL for Intel MKL-DNN.

* ANEURALNETWORKS_CONV_2D
* ANEURALNETWORKS_ADD

## Known issues
Support for Multiple Tensor inputs at runtime to model/network is ongoing

## License
Android Neural Networks HAL is distributed under the Apache License, Version 2.0
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) is an open source
performance library for Deep Learning (DL) applications intended for acceleration of DL
frameworks on Intel® architecture.


## How to provide feedback
By default, please submit an issue using native github.com interface:
https://github.com/intel/nn-hal/issues

## How to contribute

Create a pull request on github.com with your patch. Make sure your change is cleanly building
and passing ULTs.

A maintainer will contact you if there are questions or concerns.
