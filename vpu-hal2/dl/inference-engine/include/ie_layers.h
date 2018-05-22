// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief a header file for internal Layer structure to describe layers information
 * @file ie_layers.h
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "ie_common.h"
#include "ie_data.h"
#include "ie_blob.h"
#include "ie_device.hpp"
#include <map>

namespace InferenceEngine {
/**
 * @struct LayerParams
 * @brief This is an internal common Layer parameter parsing arguments
 */
struct LayerParams {
    /// @brief Layer name
    std::string name;
    /// @brief Layer type
    std::string type;
    /// @brief Layer precision
    Precision precision;
};

/**
 * @class CNNLayer
 * @brief This is a base abstract Layer - all DNN Layers inherit from this class
 */
class CNNLayer  {
public:
    /**
     * @brief A smart shared pointer
     */
    using  Ptr = std::shared_ptr<CNNLayer>;

    /**
     * @brief Layer name
     */
    std::string name;
    /**
     * @brief Layer type
     */
    std::string type;
    /**
     * @brief Layer base operating precision
     */
    Precision precision;
    /**
     * @brief A vector of pointers to the output data elements of this layer in the di-graph (order matters)
     */
    std::vector<DataPtr> outData;
    /**
     * @brief A vector of weak pointers to the input data elements of this layer in the di-graph (order matters)
     */
    std::vector<DataWeakPtr> insData;
    /**
     * @brief If suggested to fuse - a pointer to the layer which needs to be fused with this layer
     */
    Ptr _fusedWith;
    /**
     * @brief Convenience user values to store in this object as extra data
     */
    UserValue userValue;

    /**
     * @brief Layer affinity set by user.
     */
    std::string affinity;

    /**
     * @brief A constructor. Creates a new CNNLayer instance and initializes layer parameters with the given values.
     * @param prms Basic common parsing parameters
     */
    explicit CNNLayer(const LayerParams &prms) : name(prms.name), type(prms.type),
                                                 precision(prms.precision), userValue({0}) {
    }

    /**
     * @brief A virtual destructor
     */
#ifndef AKS
    virtual ~CNNLayer() {}
#else
    virtual ~CNNLayer();  //aks
#endif

    /**
     * @brief Sets a layer to be fused with
     * @param layer Reference to the layer to be fused with
     */
    void fuse(Ptr &layer) {
        _fusedWith = layer;
    }

    /**
     * @brief Returns the first element of the input data for this layer
     * @return A smart pointer to the input data element
     */
    virtual const DataPtr input() const {
        auto lockedFirstInsData = insData[0].lock();
        if (!lockedFirstInsData) {
            THROW_IE_EXCEPTION << "Internal error: unable to lock weak_ptr\n";
        }
        return lockedFirstInsData;
    }

    /**
     * @brief Checks if the input data and layer data are legitimate
     */
    virtual void validateLayer() {}


    /**
     * @brief Gets float value for the given parameter
     * @param param - name of the parameter to find
     * @param def - default value of the parameter if not found
     * @return float value
     */
    float GetParamAsFloat(const char* param, float def) {
        std::string val = GetParamAsString(param, std::to_string(def).c_str());
        try {
            return std::stof(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to float.";
        }
    }

    /**
     * @brief Returns a float value for the given layer parameter
     * @param param Name of the layer parameter
     * @return A float value for the specified parameter
     */
    float GetParamAsFloat(const char *param) {
        std::string val = GetParamAsString(param);
        try {
            return std::stof(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to float.";
        }
    }

    /**
     * @brief Returns a vector of float values for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of float values
     */
    std::vector<float> GetParamAsFloats(const char *param, std::vector<float> def) {
        std::string vals = GetParamAsString(param, "");
        std::vector<float> result;
        std::istringstream stream(vals);
        std::string str;
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stof(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                                   << ". Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of float values for the given parameter
     * @param param Name of the layer parameter
     * @return vector of float values
     */
    std::vector<float> GetParamAsFloats(const char *param) {
        std::string vals = GetParamAsString(param);
        std::vector<float> result;
        std::istringstream stream(vals);
        std::string str;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stof(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                                   << ". Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns an integer value for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return An int value for the specified parameter
     */
    int GetParamAsInt(const char *param, int def) {
        std::string val = GetParamAsString(param, std::to_string(def).c_str());
        try {
            return std::stoi(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to int.";
        }
    }

    /**
     * @brief Returns an integer value for the given parameter
     * @param param Name of the layer parameter
     * @return An int value for the specified parameter
     */
    int GetParamAsInt(const char *param) {
        std::string val = GetParamAsString(param);
        try {
            return std::stoi(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to int.";
        }
    }


    /**
     * @brief Returns a vector of int values for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of float values
     */
    std::vector<int> GetParamAsInts(const char *param, std::vector<int> def) {
        std::string vals = GetParamAsString(param, "");
        std::vector<int> result;
        std::istringstream stream(vals);
        std::string str;
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stoi(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                                   << ". Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of int values for the given parameter
     * @param param Name of the layer parameter
     * @return vector of float values
     */
    std::vector<int> GetParamAsInts(const char *param) {
        std::string vals = GetParamAsString(param);
        std::vector<int> result;
        std::istringstream stream(vals);
        std::string str;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stoi(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                                   << ". Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns a string value for the given parameter or returns the default one
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return A string value
     */
    std::string GetParamAsString(const char *param, const char *def) {
        auto it = params.find(param);
        if (it == params.end()) {
            return def;
        }
        return (*it).second;
    }

    /**
     * @brief Returns a string value for the given parameter.
     * Throws exception if parameter was not found.
     * @param param Name of the layer parameter
     * @return A string value
     */
    std::string GetParamAsString(const char *param) {
        auto it = params.find(param);
        if (it == params.end()) {
            THROW_IE_EXCEPTION << "No such parameter name '" << param << "' for layer " << name;
        }
        return (*it).second;
    }

    /**
     * @brief Map of pairs: (parameter name, parameter value)
     */
    std::map<std::string, std::string> params;
    /**
     * @brief Map of pairs: (name, weights/biases blob)
     */
    std::map<std::string, Blob::Ptr> blobs;
};

using GenericLayer = class CNNLayer;


/**
 * @class WeightableLayer
 * @brief This class represents a Layer with Weights and/or Biases (e.g. Convolution/Fully Connected, etc.)
 */
class WeightableLayer : public CNNLayer {
public:
    /**
     * @brief A default constructor. Constructs a WeightableLayer instance and initiates layer parameters with the given values
     * @param prms Initial layer parameters
     */
    explicit WeightableLayer(const LayerParams &prms) : CNNLayer(prms) {}
#ifdef AKS
    virtual ~WeightableLayer(); //aks
#endif
    /**
     * @brief A pointer to a weights blob
     */
    Blob::Ptr _weights;
    /**
     * @brief A pointer to a biases blob
     */
    Blob::Ptr _biases;
};

/**
 * @class ConvolutionLayer
 * @brief This class represents a standard 3D Convolution Layer
 */
class ConvolutionLayer : public WeightableLayer {
public:
    /**
     * @brief A convolution kernel width
     */
    unsigned int _kernel_x;
    /**
     * @brief A convolution kernel height
     */
    unsigned int _kernel_y;
    /**
     * @brief An input convolution stride width
     */
    unsigned int _stride_x;
    /**
     * @brief An Input convolution stride height
     */
    unsigned int _stride_y;
    /**
     * @brief A number of output feature maps (size) generating the 3'rd output dimension
     */
    unsigned int _out_depth;
    /**
     * @brief Input padding width
     */
    unsigned int _padding_x;
    /**
     * @brief Input padding height
     */
    unsigned int _padding_y;
    /**
     * @brief Dilation width
     */
    unsigned int _dilation_x;
    /**
     * @brief Dilation height
     */
    unsigned int _dilation_y;
    /// @brief Number of groups
    unsigned int _group;

    /**
     * @brief A default constructor. Creates a new ConvolutionLayer instance and initializes layer parameters with the given values.
     * @param prms Initial layer parameters
     */
    explicit ConvolutionLayer(const LayerParams &prms) : WeightableLayer(prms), _kernel_x(0), _kernel_y(0),
                                                         _stride_x(1), _stride_y(1), _out_depth(0),
                                                         _padding_x(0), _padding_y(0), _dilation_x(1),
                                                         _dilation_y(1), _group(1) {}
#ifdef AKS
virtual ~ConvolutionLayer();
#endif
};

/**
 * @class DeconvolutionLayer
 * @brief This class represents a standard deconvolution layer
 */
class DeconvolutionLayer : public WeightableLayer {
public:
    /**
     * @brief Convolution kernel width
     */
    unsigned int _kernel_x;
    /**
     * @brief Convolution kernel height
     */
    unsigned int _kernel_y;
    /**
     * @brief Input convolution stride width
     */
    unsigned int _stride_x;
    /**
     * @brief Input convolution stride height
     */
    unsigned int _stride_y;
    /**
     * @brief number of output feature maps (size) generating the 3'rd output dimension
     */
    unsigned int _out_depth;
    /**
     * @brief Input padding width
     */
    unsigned int _padding_x;
    /**
     * @brief Input padding height
     */
    unsigned int _padding_y;
    /**
     * @brief Dilation width
     */
    unsigned int _dilation_x;
    /**
     * @brief Dilation height
     */
    unsigned int _dilation_y;
    /**
     * @brief Number of groups
     */
    unsigned int _group;

    /**
    * @brief A default constructor. Creates a new DeconvolutionLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit DeconvolutionLayer(const LayerParams &prms) : WeightableLayer(prms), _kernel_x(0), _kernel_y(0),
                                                            _stride_x(0), _stride_y(0), _out_depth(0),
                                                            _padding_x(0), _padding_y(0), _dilation_x(0),
                                                            _dilation_y(0), _group(0) {
    }
#ifdef AKS
    virtual ~DeconvolutionLayer();
#endif
};

/**
 * @class PoolingLayer
 * @brief This class represents a standard pooling layer
 */
class PoolingLayer : public CNNLayer {
public:
    /**
     * @brief Convolution kernel width
     */
    unsigned int _kernel_x;
    /**
     * @brief Convolution kernel height
     */
    unsigned int _kernel_y;
    /**
     * @brief Input convolution stride width
     */
    unsigned int _stride_x;
    /**
     * @brief Input convolution stride height
     */
    unsigned int _stride_y;
    /**
     * @brief Input padding width
     */
    unsigned int _padding_x;
    /**
     * @brief Input padding height
     */
    unsigned int _padding_y;

    /**
     * @enum PoolType
     * @brief Defines available pooling types
     */
    enum PoolType {
        MAX = 1,
        AVG = 2,
        STOCH = 3,
        ROI = 4,
        SPACIAL_PYRAMID = 5
    };

    /**
     * @brief A pooling type
     */
    PoolType _type;

    /**
     * @brief A flag that indicates if padding is excluded or not
     */
    bool _exclude_pad;

    /**
    * @brief A default constructor. Creates a new PoolingLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit PoolingLayer(const LayerParams &prms) : CNNLayer(prms), _kernel_x(0), _kernel_y(0),
                                                     _stride_x(0), _stride_y(0),
                                                     _padding_x(0), _padding_y(0), _type(MAX), _exclude_pad(false) {}
#ifdef AKS
    virtual ~PoolingLayer();
#endif

};

/**
 * @class FullyConnectedLayer
 * @brief This class represents a fully connected layer
 */
class FullyConnectedLayer : public WeightableLayer {
public:
    /**
     * @brief A size of output
     */
    unsigned int _out_num;

    /**
    * @brief A default constructor. Creates a new FullyConnectedLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit FullyConnectedLayer(const LayerParams &prms) : WeightableLayer(prms), _out_num(0) {}
#ifdef AKS
    virtual ~FullyConnectedLayer();
#endif

};

/**
 * @class ConcatLayer
 * @brief This class represents concatenation layer
 * Takes as input several data elements and merges them to one using the supplied axis
 */
class ConcatLayer : public CNNLayer {
public:
    /**
     * @brief An axis on which concatenation operation is performed
     */
    unsigned int _axis;

    /**
    * @brief A default constructor. Creates a new ConcatLayer instance and initializes layer parameters with the given values.
    * If batch is used, then batch needs to be specified as an input dimension also
    * In current implementation 1 means channels, 0 - batch
    * @param prms Initial layer parameters
    */
    explicit ConcatLayer(const LayerParams &prms) : CNNLayer(prms), _axis(1) {}
#ifdef AKS
    virtual ~ConcatLayer();
#endif

};

/**
 * @class SplitLayer
 * @brief This class represents a layer that evenly splits the input into the supplied outputs
 */
class SplitLayer : public CNNLayer {
public:
    unsigned int _axis;

    /**
    * @brief A default constructor. Creates a new SplitLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit SplitLayer(const LayerParams &prms) : CNNLayer(prms), _axis(1) {}
#ifdef AKS
    virtual ~SplitLayer();
#endif
};

/**
 * @class NormLayer
 * @brief This class represents a Linear Response Normalization (LRN) Layer
 */
class NormLayer : public CNNLayer {
public:
    /**
     * @brief Response size
     */
    unsigned int _size;
    /**
     * @deprecated
     */
    unsigned int _k;
    /**
     * @brief Alpha coefficient
     */
    float _alpha;
    /**
     * @brief Beta coefficient
     */
    float _beta;
    /**
     * @brief Flag to specify normalization across feature maps (true) or across channels
     */
    bool _isAcrossMaps;

    /**
    * @brief A default constructor. Creates a new NormLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit NormLayer(const LayerParams &prms) : CNNLayer(prms),
                                                  _size(0), _k(1), _alpha(0), _beta(0), _isAcrossMaps(false) {}
#ifdef AKS
     virtual ~NormLayer();
#endif
};

/**
 * @class SoftMaxLayer
 * @brief This class represents standard softmax Layer
 */
class SoftMaxLayer : public CNNLayer {
public:
    /**
    * @brief A default constructor. Creates a new SoftMaxLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit SoftMaxLayer(const LayerParams &prms) : CNNLayer(prms), axis(1) {}
#ifdef AKS
    virtual ~SoftMaxLayer();
#endif

    /**
     * @brief Axis number for a softmax operation
     */
    int axis;
};

/**
 * @class ReLULayer
 * @brief This class represents a Rectified Linear activation layer
 */
class ReLULayer : public CNNLayer {
public:
    /**
    * @brief A default constructor. Creates a new ReLULayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit ReLULayer(const LayerParams &prms) : CNNLayer(prms), negative_slope(0.0f) {}
#ifdef AKS
    virtual ~ReLULayer();
#endif

    /**
     * @brief Negative slope is used to takle negative inputs instead of setting them to 0
     */
    float negative_slope;
};

#ifdef AKS
class TanHLayer : public CNNLayer {
public:
    /**
    * @brief A default constructor. Creates a new ReLULayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit TanHLayer(const LayerParams &prms) : CNNLayer(prms), negative_slope(0.0f) {}
#ifdef AKS
    virtual ~TanHLayer();
#endif

    /**
     * @brief Negative slope is used to takle negative inputs instead of setting them to 0
     */
    float negative_slope;
};

class SigmoidLayer : public CNNLayer {
public:
    /**
    * @brief A default constructor. Creates a new ReLULayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit SigmoidLayer(const LayerParams &prms) : CNNLayer(prms), negative_slope(0.0f) {}
#ifdef AKS
    virtual ~SigmoidLayer();
#endif

    /**
     * @brief Negative slope is used to takle negative inputs instead of setting them to 0
     */
    float negative_slope;
};

#endif

/**
* @class ClampLayer
* @brief This class represents a Clamp activation layer
* Clamps all tensor elements into the range [min_value, max_value]
*/
class ClampLayer : public CNNLayer {
public:
    /**
    * @brief A default constructor. Creates a new ClampLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit ClampLayer(const LayerParams &prms) : CNNLayer(prms), min_value(0.0f), max_value(1.0f) {}
#ifdef AKS
    virtual ~ClampLayer();
#endif

    /**
    * @brief A minimum value
    */
    float min_value;

    /**
    * @brief A maximum value
    */
    float max_value;
};

/**
 * @class EltwiseLayer
 * @brief This class represents an element wise operation layer
 */
class EltwiseLayer : public CNNLayer {
public:
    /**
     * @enum eOperation
     * @brief Defines possible operations that can be used
     */
    enum eOperation {
        Sum = 0, Prod, Max
    };

    /**
     * @brief A type of the operation to use
     */
    eOperation _operation;

    /**
     * @brief A vector of coefficients to scale the operands
     */
    std::vector<float> coeff;

    /**
    * @brief A default constructor. Creates a new EltwiseLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit EltwiseLayer(const LayerParams &prms) : CNNLayer(prms), _operation(Sum) {}
#ifdef AKS
    virtual ~EltwiseLayer();
#endif
};

/**
 * @class CropLayer
 * @brief This class represents a standard crop layer
 */
class CropLayer : public CNNLayer {
public:
    /**
     * @brief A vector of dimensions for cropping
     */
    std::vector<int> axis;
    /**
     * @brief A vector of dimensions to be preserved
     */
    std::vector<int> dim;
    /**
     * @brief A vector of offsets for each dimension
     */
    std::vector<int> offset;

    /**
    * @brief A default constructor. Creates a new CropLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit CropLayer(const LayerParams &prms) : CNNLayer(prms) {}
#ifdef AKS
    virtual ~CropLayer();
#endif

    /**
     * @brief Validates if crop axes match input dimensions
     * Throws an exception in case of mismatch
     */
    void validateLayer() override {
        const DataPtr in = input();
        if (axis.size() != dim.size() || axis.size() != offset.size()) {
            THROW_IE_EXCEPTION << "Incorrect format of the Crop layer.";
        }
        for (size_t i = 0; i < axis.size(); i++) {
            if (in->getDims()[axis[i]] < static_cast<size_t>(offset[i] + dim[i])) {
                THROW_IE_EXCEPTION << "Incorrect crop data! Need to crop " << dim[i] << " with offset "
                                   << offset[i] << " from input dim " << axis[i]
                                   << " with size " << in->getDims()[axis[i]];
            }
        }
    }
};

/**
 * @class ReshapeLayer
 * @brief This class represents a standard reshape layer
 */
class ReshapeLayer : public CNNLayer {
public:
    /**
     * @brief A vector of sizes of the shape
     */
    std::vector<int> shape;
    /**
     * @brief A number of axis to be taken for a reshape
     */
    int axis;
    /**
     * @brief A number of first axises to be taken for a reshape
     */
    int num_axes;

    /**
    * @brief A default constructor. Creates a new ReshapeLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit ReshapeLayer(const LayerParams &prms) : CNNLayer(prms), axis(0), num_axes(-1) {}
#ifdef AKS
    virtual ~ReshapeLayer();
#endif
};

/**
 * @class TileLayer
 * @brief This class represents a standard Tile Layer
 */
class TileLayer : public CNNLayer {
public:
    /**
     * @brief An index of the axis to tile
     */
    int axis;
    /**
     * @brief A number of copies to be made
     */
    int tiles;

    /**
    * @brief A default constructor. Creates a new TileLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit TileLayer(const LayerParams &prms) : CNNLayer(prms), axis(-1), tiles(-1) {}
#ifdef AKS
    virtual ~TileLayer();
#endif
};


/**
* @class ScaleShiftLayer
* @brief This class represents a Layer which performs Scale and Shift
*/
class ScaleShiftLayer : public WeightableLayer {
public:
    /**
     * @brief A flag that indicates if the same value is used for all the features. If false, the value is used pixel wise
     */
    unsigned int _broadcast;

public:
    /**
    * @brief A default constructor. Creates a new ScaleShiftLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit ScaleShiftLayer(const LayerParams &prms) : WeightableLayer(prms), _broadcast(0) {}
#ifdef AKS
    virtual ~ScaleShiftLayer();
#endif
};


/**
 * @class PowerLayer
 * @brief This class represents a standard Power Layer
 * Formula is: output = (offset + scale * input) ^ power
 */
class PowerLayer : public CNNLayer {
public:
    /**
     * @brief A power element
     */
    float power;
    /**
     * @brief A scale factor
     */
    float scale;
    /**
     * @brief An offset value
     */
    float offset;

    /**
    * @brief A default constructor. Creates a new PowerLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit PowerLayer(const LayerParams &prms) : CNNLayer(prms), power(1), scale(1), offset(0) {}
#ifdef AKS
    virtual ~PowerLayer();
#endif
};

/**
* @class BatchNormalizationLayer
* @brief This class represents a Batch Normalization Layer
*/
class BatchNormalizationLayer : public WeightableLayer {
public:
    /**
     * @brief A small value to add to the variance estimate to avoid division by zero
     */
    float epsilon;

public:
    /**
    * @brief A default constructor. Creates a new BatchNormalizationLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit BatchNormalizationLayer(const LayerParams &prms) : WeightableLayer(prms), epsilon(1e-3f) {}
#ifdef AKS
    virtual ~BatchNormalizationLayer();
#endif
};
}  // namespace InferenceEngine
