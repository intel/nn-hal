# Android Neural Networks HAL with OpenVINO supporting hardware accelerators such as Intel® Movidius™ NCS / MyriadX and Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)

## Introduction
The Android Neural Network Hardware Abstraction Layer (NN HAL) provides the hardware accelration for Android Neural Networks (NN) API. Intel Movidius NN HAL takes the advantage of the Intel Movidius NCS hardware & Intel MKLD-DNN , enables high performance and low power implementation of Neural Networks API. 
More information about Intel Movidius NCS is available on https://developer.movidius.com/. 
Intel MKL-DNN https://github.com/intel/mkl-dnn &  https://01.org/mkl-dnn 
Android NN API is on [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/index.html).
OpenVINO deep learning framework https://github.com/opencv/dldt & https://01.org/openvinotoolkit 


## Supported Operations
Following operations are currently supported by Android Neural Networks HAL for Intel MKL-DNN and Movidius™ NCS1.0 / MyriadX 

* ANEURALNETWORKS_CONV_2D
* ANEURALNETWORKS_DEPTHWISE_CONV_2D
* ANEURALNETWORKS_MAX_POOL_2D
* ANEURALNETWORKS_AVERAGE_POOL_2D
* ANEURALNETWORKS_FULLY_CONNECTED
* ANEURALNETWORKS_CONCATENATION
* ANEURALNETWORKS_LOGISTIC
* ANEURALNETWORKS_RELU
* ANEURALNETWORKS_RELU1
* ANEURALNETWORKS_RELU6
* ANEURALNETWORKS_TANH
* ANEURALNETWORKS_SOFTMAX
* ANEURALNETWORKS_RESHAPE
* ANEURALNETWORKS_L2_NORMALIZATION
* ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION

## Known issues
Support for Multiple Tensor inputs at runtime to model/network is ongoing   

## License
Android Neural Networks HAL is distributed under the Apache License, Version 2.0
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) is an open source performance library for Deep Learning (DL) applications intended for acceleration of DL frameworks on Intel® architecture.


## How to provide feedback
By default, please submit an issue using native github.com interface:
https://github.com/intel/nn-hal/issues

## How to contribute

Create a pull request on github.com with your patch. Make sure your change is cleanly building and passing ULTs.
A maintainer will contact you if there are questions or concerns.
