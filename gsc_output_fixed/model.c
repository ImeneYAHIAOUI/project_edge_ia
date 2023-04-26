/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "max_pooling1d_3.c" // InputLayer is excluded
#include "average_pooling1d.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_output_type dense_output) {

  // Output array allocation
  static union {
    conv1d_output_type conv1d_output;
    conv1d_1_output_type conv1d_1_output;
    conv1d_2_output_type conv1d_2_output;
    conv1d_3_output_type conv1d_3_output;
    average_pooling1d_output_type average_pooling1d_output;
    flatten_output_type flatten_output;
  } activations1;

  static union {
    max_pooling1d_output_type max_pooling1d_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    max_pooling1d_3_output_type max_pooling1d_3_output;
  } activations2;


  //static union {
//
//    static input_1_output_type input_1_output;
//
//    static conv1d_output_type conv1d_output;
//
//    static max_pooling1d_output_type max_pooling1d_output;
//
//    static conv1d_1_output_type conv1d_1_output;
//
//    static max_pooling1d_1_output_type max_pooling1d_1_output;
//
//    static conv1d_2_output_type conv1d_2_output;
//
//    static max_pooling1d_2_output_type max_pooling1d_2_output;
//
//    static conv1d_3_output_type conv1d_3_output;
//
//    static max_pooling1d_3_output_type max_pooling1d_3_output;
//
//    static average_pooling1d_output_type average_pooling1d_output;
//
//    static flatten_output_type flatten_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d(
     // First layer uses input passed as model parameter
    input,
    conv1d_kernel,
    conv1d_bias,
    activations1.conv1d_output
  );
 // InputLayer is excluded 
  max_pooling1d(
    
    activations1.conv1d_output,
    activations2.max_pooling1d_output
  );
 // InputLayer is excluded 
  conv1d_1(
    
    activations2.max_pooling1d_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations1.conv1d_1_output
  );
 // InputLayer is excluded 
  max_pooling1d_1(
    
    activations1.conv1d_1_output,
    activations2.max_pooling1d_1_output
  );
 // InputLayer is excluded 
  conv1d_2(
    
    activations2.max_pooling1d_1_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations1.conv1d_2_output
  );
 // InputLayer is excluded 
  max_pooling1d_2(
    
    activations1.conv1d_2_output,
    activations2.max_pooling1d_2_output
  );
 // InputLayer is excluded 
  conv1d_3(
    
    activations2.max_pooling1d_2_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations1.conv1d_3_output
  );
 // InputLayer is excluded 
  max_pooling1d_3(
    
    activations1.conv1d_3_output,
    activations2.max_pooling1d_3_output
  );
 // InputLayer is excluded 
  average_pooling1d(
    
    activations2.max_pooling1d_3_output,
    activations1.average_pooling1d_output
  );
 // InputLayer is excluded 
  flatten(
    
    activations1.average_pooling1d_output,
    activations1.flatten_output
  );
 // InputLayer is excluded 
  dense(
    
    activations1.flatten_output,
    dense_kernel,
    dense_bias, // Last layer uses output passed as model parameter
    dense_output
  );

}
