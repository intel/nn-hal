
#include "Blob.h"
#include<iostream>

/*
network_operations_vector get_nn_network_from_android(){

  network_operations_vector network1;
  NCSoperations operation;

  operation = RELU;
  network1.push_back(operation);
  operation = RELU6;
  network1.push_back(operation);
  operation = RELU1;
  network1.push_back(operation);
  operation = TANH;
  network1.push_back(operation);
  operation = LOGISTIC;
  network1.push_back(operation);

  return network1;
}*/


Operation_inputs_info parse_logistic_stage_info(){
  Operation_inputs_info stage1;

  stage1.main_operation = LOGISTIC;
  stage1.num_inputs = 1;
  stage1.input_shape[0] = 10;
  stage1.input_shape[1] = 1;
  stage1.input_shape[2] = 1;

  stage1.kernel_shape[0] = 0;
  stage1.kernel_shape[1] = 1;
  stage1.kernel_shape[2] = 1;

  //stage1.bias_shape = {0, 1, 1};
  stage1.output_shape[0] = 10;
  stage1.output_shape[1] = 1;
  stage1.output_shape[2] = 1;
  stage1.padding_left = 0;
  stage1.padding_right = 0;
  stage1.padding_top = 0;
  stage1.padding_bottom = 0;
  stage1.stride_width = 1;
  stage1.stride_height = 1;
  stage1.depth_multiplier = 0;
  stage1.post_operation = NONE;


  return stage1;
}

Operation_inputs_info parse_tanh_stage_info(){
  Operation_inputs_info stage1;

  stage1.main_operation = TANH;
  stage1.num_inputs = 1;
  stage1.input_shape[0] = 10;
  stage1.input_shape[1] = 1;
  stage1.input_shape[2] = 1;

  stage1.kernel_shape[0] = 0;
  stage1.kernel_shape[1] = 1;
  stage1.kernel_shape[2] = 1;

  //stage1.bias_shape = {0, 1, 1};
  stage1.output_shape[0] = 10;
  stage1.output_shape[1] = 1;
  stage1.output_shape[2] = 1;
  stage1.padding_left = 0;
  stage1.padding_right = 0;
  stage1.padding_top = 0;
  stage1.padding_bottom = 0;
  stage1.stride_width = 1;
  stage1.stride_height = 1;
  stage1.depth_multiplier = 0;
  stage1.post_operation = NONE;


  return stage1;
}

Operation_inputs_info parse_relu_stage_info(){
  Operation_inputs_info stage1;

  stage1.main_operation = RELU;
  stage1.num_inputs = 1;
  stage1.input_shape[0] = 10;
  stage1.input_shape[1] = 1;
  stage1.input_shape[2] = 1;

  stage1.kernel_shape[0] = 0;
  stage1.kernel_shape[1] = 1;
  stage1.kernel_shape[2] = 1;

  //stage1.bias_shape = {0, 1, 1};
  stage1.output_shape[0] = 10;
  stage1.output_shape[1] = 1;
  stage1.output_shape[2] = 1;
  stage1.padding_left = 0;
  stage1.padding_right = 0;
  stage1.padding_top = 0;
  stage1.padding_bottom = 0;
  stage1.stride_width = 1;
  stage1.stride_height = 1;
  stage1.depth_multiplier = 0;
  stage1.post_operation = NONE;


  return stage1;
}


Operation_inputs_info parse_relu1_stage_info(){
  Operation_inputs_info stage1;

  stage1.main_operation = RELU1;
  stage1.num_inputs = 1;
  stage1.input_shape[0] = 10;
  stage1.input_shape[1] = 1;
  stage1.input_shape[2] = 1;

  stage1.kernel_shape[0] = 0;
  stage1.kernel_shape[1] = 1;
  stage1.kernel_shape[2] = 1;

  //stage1.bias_shape = {0, 1, 1};
  stage1.output_shape[0] = 10;
  stage1.output_shape[1] = 1;
  stage1.output_shape[2] = 1;
  stage1.padding_left = 0;
  stage1.padding_right = 0;
  stage1.padding_top = 0;
  stage1.padding_bottom = 0;
  stage1.stride_width = 1;
  stage1.stride_height = 1;
  stage1.depth_multiplier = 0;
  stage1.post_operation = NONE;


  return stage1;
}

Operation_inputs_info parse_relu6_stage_info(){
  Operation_inputs_info stage1;

  stage1.main_operation = RELU6;
  stage1.num_inputs = 1;
  stage1.input_shape[0] = 10;
  stage1.input_shape[1] = 1;
  stage1.input_shape[2] = 1;

  stage1.kernel_shape[0] = 0;
  stage1.kernel_shape[1] = 1;
  stage1.kernel_shape[2] = 1;

  //stage1.bias_shape = {0, 1, 1};
  stage1.output_shape[0] = 10;
  stage1.output_shape[1] = 1;
  stage1.output_shape[2] = 1;
  stage1.padding_left = 0;
  stage1.padding_right = 0;
  stage1.padding_top = 0;
  stage1.padding_bottom = 0;
  stage1.stride_width = 1;
  stage1.stride_height = 1;
  stage1.depth_multiplier = 0;
  stage1.post_operation = NONE;


  return stage1;
}


Operation_inputs_info parse_input_stage_info(){
  Operation_inputs_info stage1;

  stage1.main_operation = INPUT;
  stage1.num_inputs = 1;
  stage1.input_shape[0] = 10;
  stage1.input_shape[1] = 1;
  stage1.input_shape[2] = 1;
  //stage1.kernel_shape = {0, 1, 1};
  //stage1.bias_shape = {0, 1, 1};
  stage1.output_shape[0] = 10;
  stage1.output_shape[1] = 1;
  stage1.output_shape[2] = 1;
  stage1.padding_left = 0;
  stage1.padding_right = 0;
  stage1.padding_top = 0;
  stage1.padding_bottom = 0;
  stage1.stride_width = 1;
  stage1.stride_height = 1;
  stage1.depth_multiplier = 0;
  stage1.post_operation = NONE;
  return stage1;
}
