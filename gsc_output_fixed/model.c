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
#include "conv1d_20.c"
#include "weights/conv1d_20.c" // InputLayer is excluded
#include "max_pooling1d_20.c" // InputLayer is excluded
#include "conv1d_21.c"
#include "weights/conv1d_21.c" // InputLayer is excluded
#include "max_pooling1d_21.c" // InputLayer is excluded
#include "conv1d_22.c"
#include "weights/conv1d_22.c" // InputLayer is excluded
#include "max_pooling1d_22.c" // InputLayer is excluded
#include "conv1d_23.c"
#include "weights/conv1d_23.c" // InputLayer is excluded
#include "max_pooling1d_23.c" // InputLayer is excluded
#include "average_pooling1d_5.c" // InputLayer is excluded
#include "flatten_5.c" // InputLayer is excluded
#include "dense_5.c"
#include "weights/dense_5.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_5_output_type dense_5_output) {

  // Output array allocation
  static union {
    conv1d_20_output_type conv1d_20_output;
    conv1d_21_output_type conv1d_21_output;
    conv1d_22_output_type conv1d_22_output;
    conv1d_23_output_type conv1d_23_output;
    average_pooling1d_5_output_type average_pooling1d_5_output;
    flatten_5_output_type flatten_5_output;
  } activations1;

  static union {
    max_pooling1d_20_output_type max_pooling1d_20_output;
    max_pooling1d_21_output_type max_pooling1d_21_output;
    max_pooling1d_22_output_type max_pooling1d_22_output;
    max_pooling1d_23_output_type max_pooling1d_23_output;
  } activations2;


  //static union {
//
//    static input_6_output_type input_6_output;
//
//    static conv1d_20_output_type conv1d_20_output;
//
//    static max_pooling1d_20_output_type max_pooling1d_20_output;
//
//    static conv1d_21_output_type conv1d_21_output;
//
//    static max_pooling1d_21_output_type max_pooling1d_21_output;
//
//    static conv1d_22_output_type conv1d_22_output;
//
//    static max_pooling1d_22_output_type max_pooling1d_22_output;
//
//    static conv1d_23_output_type conv1d_23_output;
//
//    static max_pooling1d_23_output_type max_pooling1d_23_output;
//
//    static average_pooling1d_5_output_type average_pooling1d_5_output;
//
//    static flatten_5_output_type flatten_5_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_20(
     // First layer uses input passed as model parameter
    input,
    conv1d_20_kernel,
    conv1d_20_bias,
    activations1.conv1d_20_output
  );
 // InputLayer is excluded 
  max_pooling1d_20(
    
    activations1.conv1d_20_output,
    activations2.max_pooling1d_20_output
  );
 // InputLayer is excluded 
  conv1d_21(
    
    activations2.max_pooling1d_20_output,
    conv1d_21_kernel,
    conv1d_21_bias,
    activations1.conv1d_21_output
  );
 // InputLayer is excluded 
  max_pooling1d_21(
    
    activations1.conv1d_21_output,
    activations2.max_pooling1d_21_output
  );
 // InputLayer is excluded 
  conv1d_22(
    
    activations2.max_pooling1d_21_output,
    conv1d_22_kernel,
    conv1d_22_bias,
    activations1.conv1d_22_output
  );
 // InputLayer is excluded 
  max_pooling1d_22(
    
    activations1.conv1d_22_output,
    activations2.max_pooling1d_22_output
  );
 // InputLayer is excluded 
  conv1d_23(
    
    activations2.max_pooling1d_22_output,
    conv1d_23_kernel,
    conv1d_23_bias,
    activations1.conv1d_23_output
  );
 // InputLayer is excluded 
  max_pooling1d_23(
    
    activations1.conv1d_23_output,
    activations2.max_pooling1d_23_output
  );
 // InputLayer is excluded 
  average_pooling1d_5(
    
    activations2.max_pooling1d_23_output,
    activations1.average_pooling1d_5_output
  );
 // InputLayer is excluded 
  flatten_5(
    
    activations1.average_pooling1d_5_output,
    activations1.flatten_5_output
  );
 // InputLayer is excluded 
  dense_5(
    
    activations1.flatten_5_output,
    dense_5_kernel,
    dense_5_bias, // Last layer uses output passed as model parameter
    dense_5_output
  );

}
