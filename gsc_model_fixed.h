#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>

#define FIXED_POINT	9	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-32768	// Max value for this numeric type
#define NUMBER_MAX	32767	// Min value for this numeric type
typedef int16_t number_t;		// Standard size numeric type used for weights and activations
typedef int32_t long_number_t;	// Long numeric type used for intermediate results

#ifndef min
static inline long_number_t min(long_number_t a, long_number_t b) {
	if (a <= b)
		return a;
	return b;
}
#endif

#ifndef max
static inline long_number_t max(long_number_t a, long_number_t b) {
	if (a >= b)
		return a;
	return b;
}
#endif

#if FIXED_POINT > 0 // Scaling/clamping for fixed-point representation
static inline long_number_t scale_number_t(long_number_t number) {
	return number >> FIXED_POINT;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) max(NUMBER_MIN, min(NUMBER_MAX, number));
}
#else // No scaling/clamping required for floating-point
static inline long_number_t scale_number_t(long_number_t number) {
	return number;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) number;
}
#endif


#endif //__NUMBER_H__
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      2
#define INPUT_SAMPLES       10000
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    30
#define CONV_STRIDE         10

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_4_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_4(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    2
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  30


const int16_t conv1d_4_bias[CONV_FILTERS] = {13, -4, -255, 20, -730, -377, -943, -263}
;

const int16_t conv1d_4_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-563, -172, 141, -20, -309, -38, -37, -140, -59, 11, -95, -106, -62, 236, 284, -115, -107, 140, 114, 113, 267, 114, -197, -159, 39, 61, 194, 55, -123, -276}
, {-309, -47, 600, 203, -299, -114, 84, -35, 192, 317, -73, -147, 91, 112, 189, 12, -315, -197, 108, 199, 76, -97, -263, 8, 156, -55, 38, -6, -115, -278}
}
, {{146, 102, 103, 109, 104, 84, 138, 97, 113, -74, 45, 47, 80, 137, 50, 74, -38, -21, 123, 55, 126, -32, 34, 35, 34, 85, 43, 24, 66, 195}
, {199, 66, 171, 206, 205, 162, 167, 100, 87, 45, 206, 213, 116, 138, 75, 125, 98, 116, 118, 109, 87, 121, 75, 171, 130, 102, 122, 111, 103, 285}
}
, {{-87, -7, -112, -227, -259, -123, -134, -21, -45, 17, -111, -185, -241, -126, -18, 8, -32, -244, -142, -145, -141, -33, 23, -80, -19, -106, -78, -126, -97, -27}
, {-99, -30, -49, -130, -78, -73, -133, 3, -23, 79, -15, -14, -126, -162, -127, 1, -34, -112, -33, -189, -248, -249, -142, -88, -99, -62, -46, -264, -207, -124}
}
, {{-456, -184, -132, -199, 88, 140, -29, -3, 38, -118, -69, 38, 205, 253, 200, 127, 221, 158, -65, -84, -17, -140, -28, 78, -151, -126, 2, -27, -131, -190}
, {-216, -345, -222, -283, -177, 15, 90, 298, 161, -49, 49, -14, 209, 147, 171, 152, 149, 135, 35, -19, 147, 45, -62, -225, -74, 46, -191, -60, -61, -275}
}
, {{-199, -138, -72, 32, 60, -22, -23, -22, 84, 95, -55, -118, -44, -66, 142, 93, 164, -52, -18, -57, 101, 72, 232, 108, -113, -116, -23, -86, 28, -104}
, {-306, -282, -36, 96, 18, -36, -63, -44, -48, -40, -145, -156, -155, 151, 218, 210, 145, 44, -95, -46, 51, -134, -188, -78, -55, 131, 113, 206, -65, -207}
}
, {{-97, -137, -161, -214, -300, -358, -212, -55, -98, -21, 3, -34, -86, 54, 35, -53, -105, -226, -180, -87, 49, -67, -79, -38, 31, 23, -15, -52, -150, -207}
, {14, -141, -225, -106, -11, 44, 114, 141, -34, 72, 167, 59, 115, 114, 110, 65, 92, 58, 224, 305, 275, 223, 85, 18, 102, 61, -30, -19, 23, 117}
}
, {{-253, -151, 6, 7, 165, 178, 137, 171, 150, 158, 242, 387, 291, 244, 62, -156, -257, -179, -81, 107, 135, 65, 52, 66, 130, 134, 217, 280, 228, 328}
, {-435, -122, -4, -3, 65, 17, -33, -2, -45, -76, 94, 214, 215, 154, -103, -152, -210, -204, -116, -43, 25, -5, -52, -25, -3, 30, 74, 118, 222, 320}
}
, {{-69, -128, -197, -339, -310, -265, -61, 79, 166, 73, 3, 105, 84, 17, -58, -153, -127, -120, -73, 73, 199, 192, 134, -113, -172, -216, -160, -201, -286, -419}
, {75, -112, -221, -123, -120, -146, -202, -85, -30, -6, 75, 30, 106, 92, -75, -137, -172, -258, -157, 64, 144, -119, -122, 7, 57, 55, 56, -238, -299, -239}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
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

typedef number_t max_pooling1d_4_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_4(
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
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       99
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_5_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_5(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_5_bias[CONV_FILTERS] = {-275, 121, -795, -303, -90, 116, 53, 300}
;

const int16_t conv1d_5_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-120, 34, -63}
, {70, 84, -118}
, {-85, 185, -43}
, {43, 363, 188}
, {-167, 4, -9}
, {-139, 265, 4}
, {-247, 161, 30}
, {-338, -254, -369}
}
, {{92, 140, -17}
, {-51, -580, 399}
, {65, 241, -153}
, {-78, -2, -143}
, {-116, 15, -190}
, {-15, -75, -399}
, {-300, -365, -174}
, {-9, 107, -187}
}
, {{225, -51, -112}
, {132, -35, 70}
, {-41, 53, -155}
, {-28, -141, -255}
, {305, 27, 21}
, {172, 174, 74}
, {-253, -336, -377}
, {300, 165, -175}
}
, {{-82, -240, -127}
, {22, -41, -273}
, {-410, -275, -240}
, {-103, -48, -124}
, {-288, -50, -216}
, {150, 100, -402}
, {356, 202, 80}
, {-91, -31, -146}
}
, {{-284, -101, -170}
, {-94, 92, -99}
, {-118, -253, -46}
, {-13, -158, -34}
, {-100, -215, 47}
, {-126, -17, -28}
, {-165, -60, -162}
, {-155, -48, -111}
}
, {{-172, -186, -304}
, {56, 142, 193}
, {26, 49, 47}
, {-81, -111, 18}
, {70, -84, -54}
, {183, 132, -91}
, {95, 162, 142}
, {-1, -11, -83}
}
, {{317, -102, 149}
, {-73, -280, -126}
, {-81, -555, -331}
, {171, -119, 168}
, {178, -102, -424}
, {-165, -2, 112}
, {120, -382, 217}
, {-351, 7, 64}
}
, {{32, -268, -294}
, {232, 45, -90}
, {-394, -305, -434}
, {-549, -450, -531}
, {111, 17, -59}
, {-471, -237, -279}
, {-117, -257, -837}
, {-291, -48, -44}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
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
#define INPUT_SAMPLES   97
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_5_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_5(
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
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       24
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_6_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_6(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_6_bias[CONV_FILTERS] = {-235, 38, -425, -207, -408, 205, -244, -92, -235, -185, -38, -9, 159, 176, -76, -128}
;

const int16_t conv1d_6_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-295, -56, -265}
, {-34, -79, -46}
, {-258, -61, -71}
, {-80, 87, -333}
, {-89, 64, 130}
, {-199, 56, -235}
, {-179, -229, 42}
, {-213, -268, 153}
}
, {{121, -385, -332}
, {87, -70, 320}
, {238, 60, -470}
, {0, -185, -240}
, {-86, 146, -8}
, {-147, -86, -155}
, {129, 72, -322}
, {64, -214, -589}
}
, {{-141, 12, -113}
, {-230, -113, -299}
, {-210, -147, -310}
, {67, -150, 19}
, {-59, 34, -17}
, {-54, 183, 76}
, {-136, -165, -90}
, {-132, -68, -441}
}
, {{127, -176, -375}
, {-19, 253, -18}
, {86, 54, -34}
, {-320, -115, 10}
, {-15, -290, -151}
, {83, -94, -22}
, {-79, -288, -67}
, {-474, 54, -287}
}
, {{-67, -35, -278}
, {91, 42, 321}
, {-74, 42, 131}
, {374, -435, -32}
, {23, 97, 143}
, {-22, -440, -112}
, {19, 80, 182}
, {301, 101, -285}
}
, {{25, 31, -50}
, {-61, -17, -183}
, {40, -135, -224}
, {76, 241, -14}
, {-23, -17, -121}
, {-269, -571, -178}
, {229, 166, 407}
, {126, -229, -224}
}
, {{-71, 98, -270}
, {43, 81, -421}
, {-242, -233, 181}
, {-98, -176, -134}
, {21, -9, -162}
, {-24, -229, -100}
, {-42, -75, -152}
, {-58, 165, 165}
}
, {{48, 126, 262}
, {-283, -108, 148}
, {-313, -218, -17}
, {82, -70, -239}
, {26, 68, 5}
, {113, 0, -234}
, {7, -147, 52}
, {249, -14, -188}
}
, {{-49, -11, 104}
, {-122, -245, -75}
, {-8, 120, 21}
, {175, -102, -105}
, {26, -94, 90}
, {-218, -42, -311}
, {190, 64, 410}
, {-50, 68, 26}
}
, {{-174, -125, -61}
, {-113, -174, -338}
, {-21, -60, -59}
, {-136, -260, -218}
, {58, 20, 54}
, {-263, 2, -102}
, {-76, -177, -260}
, {-233, -32, -197}
}
, {{14, -176, -148}
, {-17, 356, 151}
, {-241, -204, -212}
, {-238, -105, -99}
, {118, -68, -29}
, {-289, -225, -416}
, {-27, -25, -69}
, {219, -166, -298}
}
, {{-31, 83, -147}
, {40, -373, -256}
, {169, -402, -16}
, {114, 63, -198}
, {0, 54, -110}
, {-23, 109, -293}
, {89, -188, -399}
, {-152, -15, -294}
}
, {{-188, -127, -249}
, {-496, -406, -97}
, {-116, -164, -79}
, {-96, -261, -295}
, {-65, 16, 121}
, {29, 58, 72}
, {-214, -216, -211}
, {-383, -450, -220}
}
, {{-131, -95, -20}
, {-290, -90, -29}
, {-344, -352, -243}
, {-146, 41, -514}
, {9, -96, -90}
, {61, 81, -245}
, {0, -10, 107}
, {403, 419, 41}
}
, {{-141, -6, -116}
, {-14, -40, -154}
, {-170, -179, -105}
, {-177, -113, 18}
, {-93, 148, -54}
, {-73, 2, -145}
, {-118, 6, -72}
, {-14, -41, -50}
}
, {{-151, -150, -52}
, {-192, -251, -67}
, {-131, -15, -160}
, {18, -28, -188}
, {-62, 44, -77}
, {-386, -277, -113}
, {-125, 3, 49}
, {-95, -162, -64}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
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

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   22
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_6_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_6(
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
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       5
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_7_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_7(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_7_bias[CONV_FILTERS] = {38, 125, -125, 176, -190, 133, 182, -106, -270, -92, 205, 16, -81, 233, -399, 74, -40, 172, -10, -20, 51, -143, -18, 89, -152, -215, 92, 385, -269, 616, -209, -67}
;

const int16_t conv1d_7_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-78, 118, -85}
, {-156, 89, -287}
, {24, 5, 31}
, {-380, 15, -156}
, {-514, 71, -435}
, {-428, -144, -585}
, {-160, 403, 59}
, {-199, 17, -46}
, {-227, -51, -118}
, {84, -109, -78}
, {131, 87, -93}
, {-348, -23, 59}
, {-124, -34, 183}
, {125, -81, -38}
, {84, -19, -185}
, {-35, -73, -115}
}
, {{122, 161, -26}
, {28, -243, 22}
, {-128, -132, -70}
, {-65, -161, -278}
, {-127, -255, -351}
, {235, -218, -610}
, {-201, 20, -248}
, {127, -240, -184}
, {139, -200, -88}
, {70, 106, 122}
, {167, -129, -267}
, {-81, -411, -289}
, {55, -83, 87}
, {-142, 103, 153}
, {21, 39, 102}
, {-31, 51, -37}
}
, {{-156, -11, 36}
, {-36, 163, -181}
, {-21, -193, -146}
, {-126, -28, 76}
, {99, -30, 6}
, {-232, -442, -584}
, {34, -615, -694}
, {-30, 33, -225}
, {-343, -352, -414}
, {11, 138, -40}
, {544, 245, 264}
, {-153, -286, -330}
, {-52, 23, -264}
, {-431, 159, 389}
, {-82, 125, -75}
, {-76, -55, -163}
}
, {{-172, -184, 41}
, {-78, -143, -353}
, {-169, -207, -314}
, {-203, -17, -153}
, {41, 21, 25}
, {46, 145, 50}
, {-255, -590, -353}
, {9, 84, -133}
, {118, -261, -130}
, {153, 145, -39}
, {-125, 10, -126}
, {-501, -444, -430}
, {-97, -178, -259}
, {318, 345, 417}
, {75, -83, 134}
, {-139, -31, -70}
}
, {{134, 9, 77}
, {15, 4, -54}
, {-119, -131, -198}
, {-212, -505, -272}
, {20, -181, 76}
, {-174, -162, 0}
, {313, -132, -68}
, {-334, -96, -188}
, {-215, -178, -83}
, {61, -57, -33}
, {-125, -84, -148}
, {-248, -70, -209}
, {64, -157, -19}
, {-124, -325, -129}
, {88, -55, -17}
, {-294, 53, -71}
}
, {{-137, -43, -144}
, {-169, 80, -97}
, {-113, -41, -67}
, {-338, -256, -344}
, {-19, -211, -142}
, {-17, -231, -280}
, {-322, 29, -195}
, {-12, -90, -202}
, {-99, -27, 5}
, {123, 79, -252}
, {136, -228, -42}
, {-99, -21, -25}
, {167, -19, -86}
, {-173, -216, -237}
, {-111, -114, -69}
, {-157, -200, -61}
}
, {{-138, -82, -52}
, {-236, -308, -116}
, {-268, -109, -380}
, {-112, 157, -263}
, {-246, 32, 22}
, {67, 124, 145}
, {-175, -191, -9}
, {-2, 94, -15}
, {147, 66, 235}
, {-45, 33, -89}
, {-120, 93, -282}
, {-345, -239, -333}
, {-283, -126, -324}
, {150, -45, 326}
, {-53, -46, 10}
, {-12, -50, 116}
}
, {{116, 95, 105}
, {233, 147, 124}
, {-171, -205, -135}
, {-174, -229, 3}
, {-35, -31, -38}
, {-126, 0, 147}
, {-43, 365, -68}
, {-198, -227, -230}
, {18, -225, -43}
, {116, 52, 182}
, {452, 37, 115}
, {110, 44, 141}
, {22, -81, -58}
, {85, 0, -96}
, {25, 59, -58}
, {98, -30, -96}
}
, {{-144, 0, -21}
, {-79, -19, 30}
, {-234, -210, -53}
, {-143, -66, -34}
, {-275, 162, -112}
, {-473, -195, -495}
, {-37, -7, -56}
, {-71, -74, -167}
, {-38, -233, -204}
, {65, 38, 13}
, {0, -190, -160}
, {-215, -230, -50}
, {-169, -88, -40}
, {-480, -101, -121}
, {6, -76, 45}
, {-94, -56, 26}
}
, {{64, -18, 63}
, {-86, -113, -109}
, {-21, -1, -76}
, {-78, -34, -314}
, {12, -203, -101}
, {-177, -156, -172}
, {57, -113, 144}
, {-88, -252, -146}
, {7, -68, -189}
, {-59, 44, 45}
, {-126, -78, -20}
, {21, -117, -5}
, {-69, -183, -165}
, {-86, -64, -47}
, {24, 99, -29}
, {-44, -72, -131}
}
, {{49, 108, -99}
, {-88, 142, 199}
, {-219, -7, -158}
, {-72, 62, -26}
, {96, -258, 175}
, {253, -204, 241}
, {26, -90, 378}
, {125, 35, -129}
, {99, -141, 41}
, {0, 17, 25}
, {611, 198, 168}
, {-23, 66, -179}
, {-343, 135, -298}
, {429, 95, 239}
, {-32, 39, -12}
, {61, -113, -21}
}
, {{46, 119, -217}
, {-559, 63, -189}
, {-232, -96, -42}
, {2, -14, -36}
, {69, -282, -2}
, {118, -83, 145}
, {179, 118, -300}
, {13, -1, -168}
, {40, -276, -372}
, {39, -27, -255}
, {-228, -87, -237}
, {-182, -95, -201}
, {-88, 93, -353}
, {-102, -75, 72}
, {-29, 34, 107}
, {-2, -68, 50}
}
, {{4, 0, 69}
, {-204, -328, -149}
, {-192, -294, -161}
, {-173, -1, -163}
, {-58, -148, -6}
, {0, -61, -75}
, {205, -143, -71}
, {-267, -155, -82}
, {-85, -154, -250}
, {-67, -5, -11}
, {-191, -98, 83}
, {-148, -45, 26}
, {-253, -51, -71}
, {-64, -25, -203}
, {105, 90, 69}
, {18, 76, -43}
}
, {{-205, -276, -193}
, {-133, -233, -197}
, {157, 128, 58}
, {-140, -109, -50}
, {-950, -911, -658}
, {-231, -287, -832}
, {142, -144, 165}
, {-23, 16, -89}
, {-194, -437, -283}
, {96, 35, 183}
, {260, 336, -144}
, {-40, 49, 14}
, {131, -48, 56}
, {95, -10, -176}
, {92, -138, -69}
, {-37, -52, -65}
}
, {{46, -1, -16}
, {113, -30, -6}
, {-33, -356, -226}
, {-208, -208, -143}
, {-242, -239, -49}
, {-180, -544, -280}
, {-103, -122, -129}
, {-143, -77, -211}
, {118, -79, -59}
, {-87, 4, 46}
, {-169, -225, -286}
, {135, -198, -366}
, {14, -104, -226}
, {-489, -599, -118}
, {-72, -37, -29}
, {-111, -17, -157}
}
, {{2, -60, -66}
, {-45, -97, -161}
, {-32, 113, 12}
, {-11, -17, -199}
, {-190, -102, -421}
, {-38, -9, 49}
, {182, -254, 83}
, {174, -17, 85}
, {83, -199, -176}
, {-122, 58, 14}
, {-252, 60, 303}
, {-20, 129, 23}
, {96, -67, 18}
, {-46, 13, -303}
, {48, 5, -83}
, {77, -49, 0}
}
, {{256, -66, 201}
, {-97, -90, 14}
, {64, -162, -140}
, {-57, -308, 9}
, {-34, -344, -219}
, {-610, -597, -372}
, {20, -106, -96}
, {-114, -216, -308}
, {-301, -168, -357}
, {52, -93, 56}
, {-368, 216, -105}
, {-150, -102, -610}
, {-308, -317, -547}
, {370, -60, -433}
, {56, -125, 9}
, {13, -59, -15}
}
, {{-249, -117, -228}
, {-27, 16, -204}
, {-156, -66, -124}
, {-219, -121, -237}
, {-279, -349, -263}
, {82, -312, -261}
, {-111, -232, -190}
, {-509, 45, -91}
, {-85, -118, -183}
, {75, -148, -243}
, {373, 126, 62}
, {-169, -27, 24}
, {-232, -188, 237}
, {336, 313, -83}
, {-31, 111, -29}
, {14, -123, 17}
}
, {{92, -161, 21}
, {423, 333, 83}
, {-75, -177, -169}
, {-126, -68, -1}
, {2, -312, 88}
, {-129, -12, -7}
, {185, 23, 292}
, {-321, -69, -313}
, {-105, 98, 66}
, {15, -159, -83}
, {259, 94, 207}
, {-106, -79, -117}
, {-29, -189, -58}
, {51, -208, -276}
, {45, 67, 0}
, {-82, 9, -19}
}
, {{132, 53, -37}
, {226, 127, -453}
, {-237, 31, -230}
, {-102, -7, -298}
, {-24, 78, -128}
, {212, 290, -27}
, {-18, -87, -189}
, {-271, -37, 178}
, {80, -134, -36}
, {-39, -83, -80}
, {182, -156, -359}
, {-117, 130, -23}
, {-38, -62, -164}
, {90, -285, 313}
, {-77, -1, 99}
, {-160, 9, 194}
}
, {{-124, 30, -120}
, {-475, -475, -447}
, {5, -233, 144}
, {-474, -368, -107}
, {-359, -132, -254}
, {-217, -278, -821}
, {-123, -27, -69}
, {-115, -44, -168}
, {-112, 143, -174}
, {269, 77, 72}
, {-446, -236, -358}
, {-70, -224, -166}
, {-70, -233, 182}
, {61, 232, 129}
, {85, -14, -25}
, {-94, -148, 43}
}
, {{-12, -262, 272}
, {-84, -41, 80}
, {-211, 145, -256}
, {-268, -114, -19}
, {-279, -312, 85}
, {-274, -220, -172}
, {-97, -196, -387}
, {-159, -183, -145}
, {-86, -263, -69}
, {-107, 158, 29}
, {-81, -72, -50}
, {-315, -161, -202}
, {-154, 26, -94}
, {25, -37, -284}
, {-24, -142, -102}
, {-73, -95, -52}
}
, {{-66, 57, -37}
, {-347, 125, -81}
, {-388, 6, -194}
, {-324, -143, -233}
, {-20, 0, -520}
, {-149, 142, -173}
, {162, -2, -464}
, {-184, -149, -1}
, {-47, -94, -180}
, {-85, 12, -132}
, {182, 202, -172}
, {-303, 4, -267}
, {-260, -71, -214}
, {209, -405, 399}
, {-30, -93, 59}
, {-75, -118, -22}
}
, {{267, 27, 223}
, {-312, 285, 16}
, {-187, -127, -210}
, {-146, -144, -130}
, {-130, -97, -77}
, {271, 146, 410}
, {26, -422, 213}
, {-148, 65, -223}
, {-112, -99, -302}
, {178, 54, -140}
, {98, 158, 219}
, {-426, -397, -360}
, {-314, 47, -388}
, {374, 143, 134}
, {-40, 306, 141}
, {-44, -45, 31}
}
, {{66, 67, 100}
, {-24, -169, -243}
, {-58, -25, -221}
, {81, -156, -265}
, {-16, -329, -408}
, {-54, 98, 102}
, {539, -21, -258}
, {61, -137, -114}
, {-10, -55, -53}
, {-21, -98, 107}
, {19, 15, -151}
, {27, 12, -27}
, {287, -37, -236}
, {-48, -160, -26}
, {77, 8, 63}
, {-33, -166, -40}
}
, {{-65, 30, -84}
, {-364, -101, -367}
, {-16, 70, 11}
, {6, 347, 24}
, {-56, -232, -118}
, {-37, 76, -1}
, {-168, -58, -241}
, {-212, -151, -220}
, {-133, -91, -83}
, {-67, -136, -94}
, {-301, -175, -492}
, {-132, -191, -130}
, {-78, -311, -95}
, {62, 141, -105}
, {69, 72, -75}
, {-143, -78, -94}
}
, {{-55, -194, -155}
, {-622, -614, -314}
, {-25, -366, -38}
, {-432, -345, 188}
, {-474, -243, -155}
, {-450, -646, -417}
, {-84, 25, -117}
, {-340, -230, 29}
, {-293, -77, -199}
, {-61, -3, 149}
, {-528, -491, -186}
, {-73, -328, 44}
, {-105, -85, 217}
, {202, 404, -178}
, {-114, -38, 61}
, {104, -22, -147}
}
, {{-70, -71, -71}
, {-305, -36, -155}
, {64, -208, -125}
, {93, 76, -65}
, {-339, -33, -228}
, {-563, -375, -323}
, {-396, 156, -441}
, {-12, -19, -204}
, {-122, 114, -9}
, {-23, -71, 108}
, {-157, -220, 20}
, {-86, -43, -115}
, {216, -101, 20}
, {-344, -265, -320}
, {81, 67, -37}
, {-12, -70, 92}
}
, {{-4, -62, 36}
, {72, -423, -371}
, {77, -4, -95}
, {108, -377, -337}
, {-173, -528, -330}
, {-104, -13, -267}
, {552, -105, -223}
, {196, -355, -242}
, {-599, -371, -340}
, {-31, -91, -131}
, {-75, -209, -227}
, {204, -122, -366}
, {-54, 59, -102}
, {15, 4, -365}
, {-34, -20, 108}
, {-18, -85, 0}
}
, {{-135, 155, 46}
, {128, 122, 52}
, {-285, 32, -1}
, {75, -188, 49}
, {-175, -229, -5}
, {-104, 44, 150}
, {33, 235, 49}
, {-245, -613, -233}
, {-144, 15, -275}
, {128, -100, -147}
, {300, -101, 186}
, {-55, 0, -116}
, {306, -143, -10}
, {108, 289, -156}
, {196, 94, -111}
, {-26, -157, -143}
}
, {{-35, 42, -52}
, {-83, -99, 27}
, {-37, -168, -174}
, {-142, -257, -116}
, {-2, -118, -161}
, {53, 27, -135}
, {-82, -98, -32}
, {-103, -97, -188}
, {-216, -177, -92}
, {-26, 91, 90}
, {-34, 73, -54}
, {-199, -113, -22}
, {-67, -133, -202}
, {-290, -106, -41}
, {-40, -80, -64}
, {-45, -47, 22}
}
, {{-186, 111, -47}
, {-101, -7, -85}
, {-1, -84, -106}
, {-58, -74, -39}
, {-66, -37, -95}
, {121, -40, -69}
, {36, -9, 13}
, {-94, -11, -130}
, {1, -28, -10}
, {45, 33, -55}
, {16, -156, 56}
, {-119, 4, -18}
, {-28, -25, -65}
, {-177, -35, -45}
, {46, -49, -18}
, {41, -24, -17}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
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

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   3
#define POOL_SIZE       1
#define POOL_STRIDE     1
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_7_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_7(
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
/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   3
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t average_pooling1d_1_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_1(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned short x;
  long_number_t avg, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[k][(pos_x*POOL_STRIDE)+x];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#endif
      avg = tmp / POOL_SIZE;
      output[k][pos_x] = clamp_to_number_t(avg);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [1][32]
#define OUTPUT_DIM 32

//typedef number_t *flatten_1_output_type;
typedef number_t flatten_1_output_type[OUTPUT_DIM];

#define flatten_1 //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 32
#define FC_UNITS 3
#define ACTIVATION_LINEAR

typedef number_t dense_1_output_type[FC_UNITS];

static inline void dense_1(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 32
#define FC_UNITS 3


const int16_t dense_1_bias[FC_UNITS] = {-11, 86, -67}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{59, 145, -137, -148, -53, 72, -30, 15, 173, -140, 73, -1, 2, -65, -175, -30, 80, -196, 43, 2, 47, -63, -133, 240, -129, -89, -231, 35, -106, 79, -112, -74}
, {184, 108, -54, -219, -81, 46, -76, -151, -4, -74, -111, 70, -95, 176, -104, 108, -305, 251, -96, -51, 196, 87, -204, -345, 2, 91, 167, 53, 0, 160, -5, -115}
, {94, -45, 13, 28, -6, 62, 82, -131, -95, -150, 43, 53, -70, -4, 163, 64, -93, 51, -85, 57, -191, -135, -67, 251, -43, -126, -261, -135, -128, -250, -60, -99}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define MODEL_OUTPUT_SAMPLES 3
#define MODEL_INPUT_SAMPLES 10000 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 2

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_1_output_type dense_1_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
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
