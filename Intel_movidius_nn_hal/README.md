# Intel® Movidius™ Neural Networks HAL

## Introduction
The Intel® Movidius™ Neural Network Hardware Abstraction Layer (NN HAL) provides the hardware accelration for Android Neural Networks (NN) API. Intel Movidius NN HAL takes the advantage of the Intel Movidius NCS hardware, enables high performance and low power implementation of Neural Networks API. More information about Movidius NCS is available on [Intel® Movidius™ NCS](https://developer.movidius.com/) page and Android NN API is on [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/index.html).

## Supported Operations
Intel Movidius NCS HAL supports essential Android Neural Networks Operations.
Below are the operations supported by Intel Movidius NN HAL

* ANEURALNETWORKS_CONV_2D
* ANEURALNETWORKS_DEPTHWISE_CONV_2D
* ANEURALNETWORKS_AVERAGE_POOL_2D
* ANEURALNETWORKS_MAX_POOL_2D
* ANEURALNETWORKS_LOGISTIC
* ANEURALNETWORKS_RELU
* ANEURALNETWORKS_RELU1
* ANEURALNETWORKS_RELU6
* ANEURALNETWORKS_TANH
* ANEURALNETWORKS_SOFTMAX
* ANEURALNETWORKS_RESHAPE

## Prerequisite

[Intel® Movidius™ NCSDK](https://github.com/movidius/ncsdk)

## Integrating Intel® Movidius™ NCSDK into Intel® Movidius™ Neural Networks HAL

To integrate [Intel® Movidius™ NCSDK](https://github.com/movidius/ncsdk) into Intel® Movidius™ Neural Networks HAL follow the below steps


* Download and extract the [Intel® Movidius™ NCSDK](https://github.com/movidius/ncsdk) -version 1.12.00.01 as mentioned on the [Intel® Movidius™ NCSDK website](https://github.com/movidius/ncsdk) 
* Copy the extracted ncsdk to the specified location as shown below
```
cp <DIR>/ncsdk-1.12.00.01 <DIR>/Intel_movidius_nn_hal/libncs/ncsdk-1.12.00.01 -rf
```
* Perform the code changes to the files shown below

* Open the file **_libncs/ncsdk-1.12.00.01/api/src/mvnc_api.c_** and add a line of code after the code **// Load the mvnc executable** as shown below 
```
strcpy(mv_cmd_file, "/vendor/firmware/mvnc/MvNCAPI.mvcmd");
```
and save the changes


* Open the file **_libncs/ncsdk-1.12.00.01/api/src/usb_link_vsc.c_** and comment the header file definition or remove the line of code which contain this string **#include <sys/timeb.h>** 

* Make sure the **ncsdk-1.12.00.01** directory is located under **Intel_movidius_nn_hal/libncs**


## Validated Models
*  [Mobilenet_v1 Float paper](https://arxiv.org/pdf/1704.04861.pdf) [Mobilenet_v1 Float model](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)

## Known Issues
* After performing git clone to integrate the HAL into your Android build remove the other HAL directory using below command
```
rm -rf vpu-hal2 
```
## License
Intel® Movidius™ Neural Networks HAL is distributed under the Apache License, Version 2.0
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

## How to provide feedback
By default, please submit an issue using native github.com interface:
https://github.com/intel/nn-hal/issues

## How to contribute

Create a pull request on github.com with your patch. Make sure your change is cleanly building and passing ULTs.
A maintainer will contact you if there are questions or concerns.
