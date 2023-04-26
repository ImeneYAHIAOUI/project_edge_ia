/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   998
#define POOL_SIZE       10
#define POOL_STRIDE     10
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_12_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_12(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR