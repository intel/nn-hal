#include<stdio.h>
#include<string>
#include<iostream>
#include<stdint.h>
#include "Blob.h"


int main()
{
  network_operations_vector network1;
  NCSoperations operation;
  bool status;
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

  status = get_nn_network_from_android(network1);
  if(!status)
   return -2;

  status = prepare_blob();
  if(!status)
   return -1;

  printf("\nHELLO\n");
  return 0;
}
