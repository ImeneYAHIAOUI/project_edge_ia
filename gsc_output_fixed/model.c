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
#include "conv1d_4.c"
#include "weights/conv1d_4.c" // InputLayer is excluded
#include "max_pooling1d_4.c" // InputLayer is excluded
#include "conv1d_5.c"
#include "weights/conv1d_5.c" // InputLayer is excluded
#include "max_pooling1d_5.c" // InputLayer is excluded
#include "conv1d_6.c"
#include "weights/conv1d_6.c" // InputLayer is excluded
#include "max_pooling1d_6.c" // InputLayer is excluded
#include "conv1d_7.c"
#include "weights/conv1d_7.c" // InputLayer is excluded
#include "max_pooling1d_7.c" // InputLayer is excluded
#include "average_pooling1d_1.c" // InputLayer is excluded
#include "flatten_1.c" // InputLayer is excluded
#include "dense_1.c"
#include "weights/dense_1.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_1_output_type dense_1_output) {

  // Output array allocation
  static union {
    conv1d_4_output_type conv1d_4_output;
    conv1d_5_output_type conv1d_5_output;
    conv1d_6_output_type conv1d_6_output;
    conv1d_7_output_type conv1d_7_output;
    average_pooling1d_1_output_type average_pooling1d_1_output;
    flatten_1_output_type flatten_1_output;
  } activations1;

  static union {
    max_pooling1d_4_output_type max_pooling1d_4_output;
    max_pooling1d_5_output_type max_pooling1d_5_output;
    max_pooling1d_6_output_type max_pooling1d_6_output;
    max_pooling1d_7_output_type max_pooling1d_7_output;
  } activations2;


  //static union {
//
//    static input_2_output_type input_2_output;
//
//    static conv1d_4_output_type conv1d_4_output;
//
//    static max_pooling1d_4_output_type max_pooling1d_4_output;
//
//    static conv1d_5_output_type conv1d_5_output;
//
//    static max_pooling1d_5_output_type max_pooling1d_5_output;
//
//    static conv1d_6_output_type conv1d_6_output;
//
//    static max_pooling1d_6_output_type max_pooling1d_6_output;
//
//    static conv1d_7_output_type conv1d_7_output;
//
//    static max_pooling1d_7_output_type max_pooling1d_7_output;
//
//    static average_pooling1d_1_output_type average_pooling1d_1_output;
//
//    static flatten_1_output_type flatten_1_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_4(
     // First layer uses input passed as model parameter
    input,
    conv1d_4_kernel,
    conv1d_4_bias,
    activations1.conv1d_4_output
  );
 // InputLayer is excluded 
  max_pooling1d_4(
    
    activations1.conv1d_4_output,
    activations2.max_pooling1d_4_output
  );
 // InputLayer is excluded 
  conv1d_5(
    
    activations2.max_pooling1d_4_output,
    conv1d_5_kernel,
    conv1d_5_bias,
    activations1.conv1d_5_output
  );
 // InputLayer is excluded 
  max_pooling1d_5(
    
    activations1.conv1d_5_output,
    activations2.max_pooling1d_5_output
  );
 // InputLayer is excluded 
  conv1d_6(
    
    activations2.max_pooling1d_5_output,
    conv1d_6_kernel,
    conv1d_6_bias,
    activations1.conv1d_6_output
  );
 // InputLayer is excluded 
  max_pooling1d_6(
    
    activations1.conv1d_6_output,
    activations2.max_pooling1d_6_output
  );
 // InputLayer is excluded 
  conv1d_7(
    
    activations2.max_pooling1d_6_output,
    conv1d_7_kernel,
    conv1d_7_bias,
    activations1.conv1d_7_output
  );
 // InputLayer is excluded 
  max_pooling1d_7(
    
    activations1.conv1d_7_output,
    activations2.max_pooling1d_7_output
  );
 // InputLayer is excluded 
  average_pooling1d_1(
    
    activations2.max_pooling1d_7_output,
    activations1.average_pooling1d_1_output
  );
 // InputLayer is excluded 
  flatten_1(
    
    activations1.average_pooling1d_1_output,
    activations1.flatten_1_output
  );
 // InputLayer is excluded 
  dense_1(
    
    activations1.flatten_1_output,
    dense_1_kernel,
    dense_1_bias, // Last layer uses output passed as model parameter
    dense_1_output
  );

}
