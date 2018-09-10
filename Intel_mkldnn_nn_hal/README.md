# Intel® MKL-DNN Neural Networks HAL

## Introduction
The Intel® MKL-DNN(Intel® Math Kernel Library for Deep Neural Networks) Neural Network Hardware Abstraction Layer (NN HAL) provides the hardware accelration for Android Neural Networks (NN) API. Intel MKL-DNN Neural Networks HAL takes the advantage of the Intel® MKL-DNN, enables high performance and low power implementation of Neural Networks API. More information about Intel® MKL-DNNis available on [Intel® MKL-DNN](https://github.com/intel/mkl-dnn).

## Supported Operations
Intel MKL-DNN Neural Networks HAL supports essential Android Neural Networks Operations.
Below are the operations supported by Intel MKL-DNN Neural Networks HAL

* ANEURALNETWORKS_CONV_2D
* ANEURALNETWORKS_DEPTHWISE_CONV_2D
* ANEURALNETWORKS_FULLY_CONNECTED
* ANEURALNETWORKS_AVERAGE_POOL_2D
* ANEURALNETWORKS_MAX_POOL_2D
* ANEURALNETWORKS_ADD
* ANEURALNETWORKS_LOGISTIC
* ANEURALNETWORKS_RELU
* ANEURALNETWORKS_RELU6
* ANEURALNETWORKS_TANH
* ANEURALNETWORKS_SOFTMAX
* ANEURALNETWORKS_CONCATENATION

## Prerequisite
[Intel® MKL-DNN](https://github.com/intel/mkl-dnn).

## Integrating Intel® MKL-DNN into Intel® MKL-DNN Neural Networks HAL

To integrate [Intel® MKL-DNN](https://github.com/intel/mkl-dnn) into Intel® MKL-DNN Neural Networks HAL follow the below steps


* Download and extract the [Intel® MKL-DNN](https://github.com/intel/mkl-dnn) -v0.14 as mentioned on the [Intel® MKL-DNN](https://github.com/intel/mkl-dnn)
* Copy the extracted ncsdk to the specified location as shown below
```
cp <DIR>/mkl-dnn <DIR>/Intel_movidius_nn_hal/libmkldnn/mkl-dnn -rf
```
* Make sure the **mkl-dnn** directory is located under **Intel_mkldnn_nn_hal/libmkldnn**


## Validated Models
*  [Mobilenet_v1 Float paper](https://arxiv.org/pdf/1704.04861.pdf) [Mobilenet_v1 Float model](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)

## Known Issues
* Quant is not supported.
* Do not support beta!=1.0 in operation ANEURALNETWORKS_SOFTMAX
* Do not support dim=3 in operation ANEURALNETWORKS_CONCATENATION

## License
Intel® MKL-DNN Neural Networks HAL is distributed under the Apache License, Version 2.0
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

## How to provide feedback
By default, please submit an issue using native github.com interface:
https://github.com/intel/nn-hal/issues

## How to contribute

Create a pull request on github.com with your patch. Make sure your change is cleanly building and passing ULTs.
A maintainer will contact you if there are questions or concerns.
