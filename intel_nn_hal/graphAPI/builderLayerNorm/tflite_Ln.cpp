#include "iostream"
#include "tflite_Ln.h"

void PortableMeanStddevNormalization(const float* input_vector,
                                     float* output_vector, int v_size,
                                     int n_batch, float normalization_epsilon) {
  for (int batch = 0; batch < n_batch; ++batch) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < v_size; ++i) {
      sum += input_vector[i];
      sum_sq += input_vector[i] * input_vector[i];
    }
    const float mean = sum / v_size;
    float stddev_inv = 0.0f;
    const float variance = sum_sq / v_size - mean * mean;
    if (variance == 0) {
    std::cout << "variance = 0\n";
      stddev_inv = 1.0f / std::sqrt(normalization_epsilon);
    } else {
      stddev_inv = 1.0f / std::sqrt(variance);
    }
    for (int i = 0; i < v_size; ++i) {
      output_vector[i] = (input_vector[i] - mean) * stddev_inv;
    }
    input_vector += v_size;
    output_vector += v_size;
  }
}