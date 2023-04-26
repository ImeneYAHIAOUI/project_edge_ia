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

typedef number_t conv1d_24_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_24(
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


const int16_t conv1d_24_bias[CONV_FILTERS] = {-61, -52, 25, -272, -109, -4, -498, -89}
;

const int16_t conv1d_24_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{98, 116, -42, 41, -1, -29, -16, 128, 94, -13, -77, 33, 64, -35, -60, 39, -41, 67, 42, 151, 106, 4, -38, -70, 12, 3, -1, -1, 82, 111}
, {358, 206, 13, 13, 44, -153, -244, -3, 41, 64, 107, 247, 81, 148, 7, -143, -98, -136, -9, -43, 19, -65, 29, 108, 60, -80, -96, -227, -106, 2}
}
, {{-95, -52, 127, 224, 203, 149, -29, -94, 86, 145, -31, 100, 221, 134, 155, 396, 0, -216, -163, -198, -40, 297, 110, 15, 215, -53, -258, 31, -1, -184}
, {12, -68, 64, 258, 89, 11, 102, -77, 40, 73, -80, 30, 19, -35, 18, 260, 31, -99, -26, -37, 14, 159, 21, -18, 92, -50, -239, -61, -71, -12}
}
, {{348, 134, 96, 51, -63, 27, -31, 133, 198, 153, 61, 136, 193, 216, 172, -23, -59, -160, -36, -10, -66, 70, 261, 197, 131, 180, 186, 53, -102, -168}
, {209, -35, -119, -80, -134, -89, 33, -92, -106, -17, 86, 121, 89, -3, -55, -177, -100, -158, -62, -121, -134, -44, -4, 80, -30, 126, 111, 58, -29, -188}
}
, {{-176, -135, -158, -79, -126, -82, -118, -128, -73, -53, -134, -49, -46, -5, -97, -24, -98, -44, -129, -121, -84, -106, -50, -16, 17, 37, 37, -1, -47, -77}
, {111, 151, 138, 88, 13, 4, 19, 76, 34, 48, 114, -2, 33, 46, -4, 47, 30, 6, -44, -54, -89, 0, 30, 106, 122, 125, 140, 89, 49, -2}
}
, {{-200, -195, -149, -65, -73, -46, 29, -161, -168, -23, 92, 71, 186, 120, -20, 97, 217, 42, 49, 149, -17, 10, -30, -106, -101, -1, 62, -62, -39, -131}
, {-197, -148, -9, -99, -78, -1, 36, -65, -24, -7, 147, 62, 188, 123, -56, 44, -10, -10, 13, 5, 32, -28, 49, -31, -132, -69, -109, -146, -127, -139}
}
, {{124, 39, 137, 157, 213, 141, 119, 54, 141, 132, 109, 46, 64, 40, 7, 65, 87, 73, 131, 133, 128, 157, 150, 119, 190, 127, 81, 6, 80, 94}
, {178, 112, 98, 128, 101, 135, 117, 107, 177, 68, 89, 52, 69, 104, 84, 161, 23, 63, 9, 181, 174, 243, 149, 161, 124, 139, 99, 152, 143, 234}
}
, {{75, 97, 207, 169, 225, 245, 136, 267, 177, 150, 64, 112, 40, 129, 39, 168, 36, 98, 92, 34, 19, 6, -14, -103, -62, -27, -194, -175, -169, -191}
, {167, 145, 151, 114, 40, 85, 58, 131, 86, 174, 47, -24, 98, 40, 61, 133, 121, 31, 42, -63, -18, -86, -159, -197, -141, -149, -205, -214, -194, -163}
}
, {{-108, -72, -36, -48, 61, 51, -15, 2, -160, -157, -124, 3, -4, 105, 71, -24, -68, -59, -107, -53, -89, -20, -73, -72, -100, -19, -60, -100, -55, -57}
, {-211, -192, -116, -59, 40, 92, -3, -71, -67, -121, -142, -33, 27, 15, 15, -27, -128, -124, -118, -124, -155, -144, -37, -67, -57, -111, -111, -29, -47, -4}
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

typedef number_t max_pooling1d_24_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_24(
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

typedef number_t conv1d_25_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_25(
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


const int16_t conv1d_25_bias[CONV_FILTERS] = {-89, -198, 42, -121, 208, 158, 92, -121}
;

const int16_t conv1d_25_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-62, -316, -232}
, {-211, 19, -232}
, {-338, -221, -173}
, {164, 66, 1}
, {74, -56, -224}
, {-89, 105, 155}
, {69, -100, -181}
, {4, -32, -185}
}
, {{-145, -151, -141}
, {-130, 10, -81}
, {-120, -159, -188}
, {-98, -6, 37}
, {88, 55, -50}
, {16, -25, 17}
, {120, 35, 239}
, {82, 121, 127}
}
, {{-158, 106, 201}
, {57, -112, 134}
, {-24, 90, 243}
, {-60, -60, 31}
, {-257, -153, 2}
, {197, -209, -315}
, {125, -124, -157}
, {-242, 102, -59}
}
, {{5, 11, 127}
, {251, -17, 240}
, {-37, -362, -91}
, {-64, -144, 117}
, {137, 37, -132}
, {-220, -127, 139}
, {-50, -286, 209}
, {108, -44, 72}
}
, {{-58, -228, -135}
, {-73, -269, -263}
, {-248, -273, -87}
, {-117, 119, 162}
, {-209, -50, 57}
, {44, -60, 296}
, {-248, -63, -470}
, {125, 260, 150}
}
, {{-55, -137, -9}
, {-129, -46, -92}
, {-26, 54, 103}
, {-365, -252, -297}
, {-46, 47, 141}
, {-144, 257, 102}
, {-41, -250, -249}
, {-51, -9, -131}
}
, {{119, -228, 27}
, {-73, -136, 78}
, {64, -61, 55}
, {29, -276, -139}
, {74, -195, -231}
, {-141, 117, 112}
, {-327, -213, -349}
, {27, -2, -191}
}
, {{30, -12, -7}
, {-136, -31, -150}
, {32, 72, 151}
, {135, 157, -58}
, {-104, -166, -110}
, {134, 175, 182}
, {-73, -28, 46}
, {138, -5, 106}
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

typedef number_t max_pooling1d_25_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_25(
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

typedef number_t conv1d_26_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_26(
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


const int16_t conv1d_26_bias[CONV_FILTERS] = {157, 53, -287, 58, -170, 124, -212, -501, -4, -106, 195, -162, -195, -127, 290, 260}
;

const int16_t conv1d_26_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-80, 183, 93}
, {147, -97, 104}
, {-211, -237, -71}
, {18, 69, 186}
, {-424, -365, -39}
, {-165, -162, -156}
, {-203, -223, -138}
, {-99, -86, 64}
}
, {{50, -162, 68}
, {-21, -6, 187}
, {-46, -209, 68}
, {4, 63, 48}
, {-116, 143, 177}
, {-22, -151, -207}
, {113, -208, -135}
, {135, -238, -240}
}
, {{64, -297, 73}
, {-157, 28, 20}
, {125, 106, -313}
, {-368, 142, 23}
, {-117, -107, -134}
, {-99, 66, -77}
, {1, -15, 26}
, {161, -134, 30}
}
, {{-241, -166, -48}
, {-22, -41, -29}
, {-137, -238, -341}
, {-119, -229, -92}
, {-92, -156, -182}
, {-56, -246, -37}
, {-88, -7, 38}
, {-101, 22, -77}
}
, {{65, -54, 86}
, {-304, 40, 5}
, {-120, -69, 224}
, {-298, -229, -262}
, {-62, -51, 223}
, {-395, -102, -65}
, {-84, -42, -194}
, {-182, -64, -16}
}
, {{-130, -246, -129}
, {-131, -183, -26}
, {258, -150, -318}
, {92, -163, -217}
, {0, -258, -238}
, {-124, -22, -2}
, {73, 33, -107}
, {57, -60, -192}
}
, {{-23, -117, -9}
, {-76, -134, 116}
, {-170, -182, 20}
, {-3, -24, -44}
, {-204, 24, 126}
, {-103, 133, -183}
, {18, -21, -12}
, {-264, -107, -27}
}
, {{99, 43, 4}
, {16, -5, 46}
, {79, 228, -12}
, {79, 60, 34}
, {68, 111, 301}
, {-158, -204, -173}
, {-58, -190, -71}
, {-60, -57, -3}
}
, {{-310, -393, -42}
, {-20, -112, -110}
, {-173, -189, -35}
, {-332, -484, -443}
, {-58, -252, 6}
, {-57, 86, -53}
, {9, 24, 45}
, {-43, -104, 33}
}
, {{-53, -119, -94}
, {-266, -153, 31}
, {16, -42, -300}
, {34, -189, -123}
, {-113, 23, -22}
, {-260, -84, -153}
, {-189, -66, -81}
, {-137, -13, -37}
}
, {{-142, -340, 85}
, {160, -80, -23}
, {-336, -431, -167}
, {-90, -244, -217}
, {-260, -314, -252}
, {100, -108, 72}
, {-84, -148, 17}
, {-22, 57, 58}
}
, {{-215, -87, -136}
, {-150, -7, 21}
, {-113, 34, -289}
, {63, -227, -195}
, {21, -78, -98}
, {-137, 27, 25}
, {85, 39, 40}
, {-156, -265, -123}
}
, {{-188, -89, 81}
, {-44, -194, -10}
, {168, -202, -128}
, {19, -177, -91}
, {-194, -44, -106}
, {-22, 82, -183}
, {-70, -77, -49}
, {-91, 48, -25}
}
, {{-23, -49, -5}
, {130, 100, 113}
, {-28, -13, -209}
, {-8, -92, -15}
, {-76, -238, 121}
, {-339, -191, -127}
, {-263, -283, -91}
, {46, 150, 144}
}
, {{4, -179, -132}
, {-111, -190, -97}
, {18, 24, -135}
, {-23, -48, 26}
, {-94, -323, -506}
, {77, -44, -123}
, {-39, -25, -39}
, {-367, -234, -155}
}
, {{-90, -178, -40}
, {-76, -79, -91}
, {-49, 157, 9}
, {-378, -193, 147}
, {-405, -491, -209}
, {154, -173, -140}
, {-102, 111, -46}
, {-117, -127, -165}
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

typedef number_t max_pooling1d_26_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_26(
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

typedef number_t conv1d_27_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_27(
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


const int16_t conv1d_27_bias[CONV_FILTERS] = {148, -47, -38, -38, 355, 451, 139, -5, 136, 149, 304, 168, 272, -191, -66, 358, 129, 8, 110, 18, 72, 57, -100, -46, -33, 256, -149, -112, 46, -5, 217, -26}
;

const int16_t conv1d_27_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-309, -229, -39}
, {-105, -124, -113}
, {63, -80, -59}
, {0, -53, 71}
, {-36, -48, -131}
, {-77, -75, -303}
, {40, -111, -35}
, {-202, -205, -327}
, {36, 164, -45}
, {-96, -95, 107}
, {16, 9, 232}
, {-73, -103, -126}
, {78, 202, -24}
, {-57, -28, 140}
, {-65, -100, -134}
, {-141, -331, -207}
}
, {{-166, -60, -303}
, {-14, -30, -51}
, {46, 57, 96}
, {-66, -134, -72}
, {-138, -101, 250}
, {174, 108, 126}
, {-6, -51, -26}
, {-25, -36, 12}
, {-52, 62, -17}
, {134, 98, 90}
, {-265, -355, -197}
, {-104, 41, 83}
, {-33, 73, 89}
, {34, -144, -91}
, {65, 204, 90}
, {65, 225, 198}
}
, {{-197, -68, 103}
, {4, -43, 27}
, {44, 155, -54}
, {125, -11, 71}
, {-151, -87, 4}
, {15, -113, -297}
, {22, -82, 9}
, {-362, -146, -205}
, {96, 58, 168}
, {39, -34, -34}
, {-69, 151, 219}
, {61, -168, -162}
, {28, 127, -19}
, {-216, -32, 149}
, {131, -7, -2}
, {-89, -272, 71}
}
, {{-64, -124, -20}
, {-184, -131, -33}
, {38, -84, 1}
, {34, 43, 89}
, {-59, -4, -66}
, {16, 23, -43}
, {-64, 54, 52}
, {-37, 6, -15}
, {58, 29, -133}
, {-35, -119, -124}
, {-7, 11, -26}
, {-69, -16, -126}
, {0, -113, -33}
, {-96, 29, -137}
, {-150, -32, -80}
, {-113, -40, 1}
}
, {{-41, 34, -129}
, {102, -3, 14}
, {-159, -48, -196}
, {81, 21, -113}
, {-158, -39, -358}
, {-337, 23, 110}
, {-48, -100, 32}
, {26, 104, -99}
, {-78, 204, 82}
, {35, -11, -104}
, {83, 100, 18}
, {-18, 148, -115}
, {-77, -68, -198}
, {-239, 116, -325}
, {167, 132, 269}
, {-32, -149, -5}
}
, {{-188, -273, -13}
, {-190, -216, -213}
, {-3, -66, -16}
, {67, -77, 45}
, {-169, -79, 146}
, {-55, 220, -95}
, {-111, 104, -114}
, {-95, -75, 63}
, {8, 90, 120}
, {93, -48, 65}
, {84, -38, 27}
, {-50, -45, 57}
, {22, -45, 79}
, {127, -100, 112}
, {-152, 74, -64}
, {-196, -216, -204}
}
, {{8, 113, 113}
, {-147, -100, -19}
, {24, 57, -353}
, {-81, -5, -128}
, {85, 117, 74}
, {-51, 155, -61}
, {-149, 71, 10}
, {-130, -45, -152}
, {18, -80, -144}
, {-131, -84, 20}
, {19, 85, -14}
, {4, -113, -141}
, {79, 181, -16}
, {-101, -79, -187}
, {224, 103, 249}
, {185, 108, 170}
}
, {{50, -74, 12}
, {-17, -117, -100}
, {-10, 68, -124}
, {51, -48, 18}
, {-153, -204, 81}
, {174, 83, 106}
, {-66, 65, -15}
, {-116, 63, -63}
, {1, -7, -69}
, {34, 40, -113}
, {68, -31, 41}
, {59, -19, 88}
, {0, 143, -104}
, {25, -15, -171}
, {-73, 22, -2}
, {83, 58, 394}
}
, {{-77, -46, 187}
, {-142, -92, -82}
, {-12, -34, 42}
, {4, -3, 2}
, {-70, -4, 35}
, {-102, -127, -16}
, {-88, -83, -27}
, {-221, -173, -141}
, {44, 227, 50}
, {-99, -46, 97}
, {33, -153, -41}
, {30, -54, 16}
, {-22, -58, -169}
, {-12, -278, 71}
, {80, 26, -423}
, {-146, -284, -343}
}
, {{-96, -40, -185}
, {-420, -316, -386}
, {-145, -62, -141}
, {-91, -60, -61}
, {14, 28, -104}
, {-90, -6, -150}
, {-49, 11, -107}
, {66, -154, 4}
, {94, 200, 49}
, {73, -70, -139}
, {-203, -676, -229}
, {99, -96, 43}
, {37, 284, -94}
, {39, -116, -110}
, {-120, 213, 243}
, {-81, 127, 182}
}
, {{-118, -258, -20}
, {-133, -227, -82}
, {73, -177, -30}
, {-78, 63, -78}
, {-199, 247, -150}
, {80, 97, 67}
, {-42, 93, 23}
, {-128, -85, -65}
, {126, 56, 4}
, {-69, -131, -35}
, {-97, -373, -73}
, {-143, -30, -74}
, {-193, 43, 2}
, {-61, -203, 141}
, {-369, -240, -281}
, {-236, -276, 16}
}
, {{-38, -113, -314}
, {25, -202, -287}
, {-223, -79, -44}
, {95, 152, -18}
, {43, -223, -50}
, {-292, -173, -292}
, {151, -77, -174}
, {-177, -237, -170}
, {125, 214, 339}
, {-7, -104, -58}
, {199, -103, -284}
, {137, 219, 35}
, {-129, -91, 78}
, {131, -109, -278}
, {91, -399, -252}
, {-204, -517, -136}
}
, {{-37, -12, -515}
, {-95, -55, 16}
, {-245, -78, -185}
, {-163, -200, -90}
, {-196, -114, 93}
, {-108, 28, -15}
, {-11, -53, -88}
, {-121, -58, -158}
, {-38, 20, -104}
, {-12, 37, -164}
, {-213, -186, -155}
, {9, -114, -70}
, {152, 68, 6}
, {-260, 79, -49}
, {130, 267, 133}
, {50, 167, 222}
}
, {{-63, -1, -44}
, {-104, -79, -142}
, {67, -23, -34}
, {-138, -156, -197}
, {76, 314, 198}
, {-138, -147, -268}
, {-16, -122, -45}
, {-92, -100, -48}
, {-105, -18, -64}
, {-55, -2, 32}
, {-63, -170, -213}
, {-50, 67, 46}
, {-124, -114, -241}
, {23, -95, -80}
, {-68, -22, -10}
, {-83, -231, -114}
}
, {{29, -172, -66}
, {-66, 112, -97}
, {-194, -3, -96}
, {62, 100, -103}
, {-215, -94, 14}
, {-53, -102, -198}
, {-39, -57, -7}
, {-79, -135, -180}
, {185, -51, 12}
, {-55, -42, 16}
, {-49, 81, -47}
, {9, 1, -49}
, {-10, -63, 98}
, {-88, -170, -152}
, {-132, 31, -177}
, {-13, 28, -26}
}
, {{145, 106, 194}
, {101, 126, 105}
, {13, -209, -176}
, {84, -70, 105}
, {-73, -195, 74}
, {84, 35, -172}
, {156, -66, -9}
, {103, -123, 55}
, {16, -89, 114}
, {19, -179, -77}
, {-57, 8, 23}
, {39, 210, -6}
, {-7, -183, -4}
, {-197, -9, -173}
, {179, 22, 172}
, {64, 0, -43}
}
, {{-303, -142, -248}
, {-194, -87, -151}
, {-58, 148, -85}
, {-30, -19, -76}
, {299, 32, 300}
, {145, 199, 43}
, {-78, -4, 21}
, {-103, -222, 84}
, {95, 94, -101}
, {-175, -142, -147}
, {-190, -257, -316}
, {-151, 153, 101}
, {253, 69, 352}
, {-86, -154, -6}
, {-110, 71, 134}
, {-379, 12, 68}
}
, {{-12, 33, -102}
, {-152, -151, -86}
, {-62, 26, -32}
, {81, 45, -76}
, {-115, -94, 49}
, {-154, -21, -20}
, {-127, -108, -14}
, {-79, -58, -59}
, {6, -6, 118}
, {-86, -57, 69}
, {16, -47, 74}
, {46, -99, 31}
, {25, -9, 42}
, {-37, -71, -38}
, {79, 132, 33}
, {16, -208, -365}
}
, {{-64, -8, -78}
, {-29, -138, 12}
, {-165, 105, -160}
, {61, -48, -91}
, {-92, 6, -152}
, {278, 15, 80}
, {-132, -55, -43}
, {-21, -99, 74}
, {-74, 75, 17}
, {38, 50, -7}
, {-107, 88, 1}
, {69, -117, -36}
, {-102, -36, 100}
, {-62, -44, -138}
, {-34, -3, 241}
, {-57, 67, 284}
}
, {{-155, -129, 16}
, {-110, -129, -60}
, {12, 47, -64}
, {85, 36, -22}
, {-214, -96, -103}
, {-47, -164, -186}
, {-17, 41, -90}
, {-89, -151, -81}
, {130, 161, 113}
, {-18, 68, -7}
, {-70, -62, 75}
, {-122, 125, -75}
, {80, -62, -74}
, {-172, -85, 49}
, {-2, 99, -42}
, {61, -223, -100}
}
, {{115, 87, 237}
, {30, 81, 119}
, {-64, -73, -95}
, {37, 34, -81}
, {-244, -29, -83}
, {-165, -47, -153}
, {-70, -4, -23}
, {-185, -115, -111}
, {-90, -62, 32}
, {48, -19, -24}
, {-27, -69, -101}
, {-66, 40, 8}
, {-8, -161, -121}
, {-30, 50, -116}
, {-32, -13, 59}
, {-110, -39, -83}
}
, {{-47, -84, 143}
, {-123, -76, -37}
, {-208, -172, -139}
, {-110, -135, -135}
, {-132, -169, -223}
, {-174, -63, -259}
, {-168, -196, -30}
, {-125, -83, -103}
, {74, 110, 122}
, {26, -109, 21}
, {-87, -98, 129}
, {114, -55, -231}
, {62, 5, 33}
, {-65, -249, 55}
, {183, -10, -60}
, {-40, -181, -119}
}
, {{25, -160, 26}
, {-169, -81, -111}
, {-61, -116, -11}
, {-102, -78, -79}
, {-244, -114, 7}
, {-85, 53, -9}
, {79, 55, -31}
, {-203, -113, -93}
, {-137, -101, -17}
, {-141, -100, -12}
, {19, -41, 0}
, {78, 4, -69}
, {-57, 7, 59}
, {-119, -118, -227}
, {-139, -99, -107}
, {-132, -45, -118}
}
, {{88, -38, -162}
, {-88, -9, -60}
, {79, -125, -60}
, {-44, -98, 73}
, {-212, 217, 76}
, {43, 70, -21}
, {-129, 111, 70}
, {-121, -120, -34}
, {-93, -31, -28}
, {-4, -58, 33}
, {-16, 129, 109}
, {81, 66, 70}
, {-116, 148, 143}
, {164, -27, 30}
, {-153, 22, 92}
, {25, 95, -3}
}
, {{-101, -146, -123}
, {-139, -70, -210}
, {32, -10, -152}
, {47, 51, -2}
, {10, -45, 31}
, {-40, 55, 42}
, {-129, 46, -69}
, {-73, -135, -103}
, {123, 22, 210}
, {-163, -175, -62}
, {17, -34, 2}
, {-19, -144, -36}
, {-15, 38, 68}
, {-81, -85, -169}
, {-216, -133, 0}
, {-45, -111, 126}
}
, {{126, -105, -26}
, {146, 107, 32}
, {-190, -371, -51}
, {68, 73, -63}
, {95, -429, -130}
, {-290, -140, 26}
, {-139, -377, -112}
, {-30, -8, 69}
, {88, -82, -24}
, {-135, -156, -67}
, {100, 8, 46}
, {-40, -38, 53}
, {163, -4, -57}
, {-279, -365, -481}
, {302, 56, 367}
, {115, -25, 152}
}
, {{239, -21, 46}
, {-15, -208, -92}
, {7, -43, -108}
, {-11, -44, 74}
, {-220, 1, -19}
, {-104, -274, -133}
, {-108, 34, 1}
, {-123, -141, -184}
, {-131, -65, -43}
, {-27, 66, 81}
, {100, 125, -26}
, {-35, -19, 14}
, {68, -175, 137}
, {218, -111, -106}
, {-53, -93, 60}
, {118, -30, -132}
}
, {{-45, 6, -110}
, {-61, 10, -41}
, {-49, -57, -32}
, {-33, 25, -68}
, {44, -58, -139}
, {-56, -131, -71}
, {-46, 50, 0}
, {-134, -73, -22}
, {19, -129, 29}
, {-80, -8, -2}
, {-21, -92, 11}
, {56, 38, -52}
, {-41, 85, 101}
, {-27, -114, -37}
, {-75, -166, -174}
, {-14, -19, -51}
}
, {{-28, -70, -89}
, {-149, -149, -147}
, {100, -14, 128}
, {50, -68, 78}
, {-35, 84, -80}
, {-148, -205, -181}
, {0, -69, 72}
, {28, -122, -178}
, {75, 104, 141}
, {108, -39, -9}
, {129, 31, 42}
, {0, -140, -102}
, {-8, -132, -61}
, {161, -15, -140}
, {15, -117, -56}
, {157, 18, -185}
}
, {{3, -33, -5}
, {-27, -48, -23}
, {20, -19, 6}
, {12, -158, 37}
, {-188, 39, 18}
, {143, -97, 62}
, {-28, 23, -7}
, {-10, -10, -100}
, {-145, -130, -6}
, {-50, 11, 36}
, {20, -38, -15}
, {46, -112, -93}
, {49, -153, -25}
, {-101, -47, -97}
, {-62, 91, 115}
, {-4, 38, 176}
}
, {{11, -185, -86}
, {-165, 12, 1}
, {113, -21, 36}
, {-61, -77, -82}
, {-279, 103, 256}
, {445, 118, 280}
, {21, -282, 35}
, {0, -68, 144}
, {-18, 21, -146}
, {-1, 43, -42}
, {-52, -72, -119}
, {6, -210, 104}
, {224, 50, 84}
, {-37, -17, -130}
, {45, 11, 46}
, {125, 266, 456}
}
, {{30, -100, 52}
, {-77, -14, -45}
, {22, -101, -128}
, {21, -45, 14}
, {81, -63, -131}
, {10, -227, -23}
, {6, -41, -88}
, {-119, -56, -77}
, {35, -47, -139}
, {-7, -78, -77}
, {-26, 63, 38}
, {24, -111, -150}
, {166, 130, -161}
, {-69, -22, -45}
, {25, 2, -81}
, {46, -137, -179}
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

typedef number_t max_pooling1d_27_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_27(
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

typedef number_t average_pooling1d_6_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_6(
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

//typedef number_t *flatten_6_output_type;
typedef number_t flatten_6_output_type[OUTPUT_DIM];

#define flatten_6 //noop (IN, OUT)  OUT = (number_t*)IN

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

typedef number_t dense_6_output_type[FC_UNITS];

static inline void dense_6(
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


const int16_t dense_6_bias[FC_UNITS] = {7, -53, 40}
;

const int16_t dense_6_kernel[FC_UNITS][INPUT_SAMPLES] = {{-81, 95, -127, -79, -42, 44, 74, 63, -21, 117, 20, -347, 236, -88, -80, 20, -39, -71, 45, 29, -142, 26, -166, -3, 12, -8, -81, -142, 9, 5, 118, -154}
, {96, -33, 99, -40, -269, 120, -237, -29, 89, 114, 51, 319, -371, -157, 25, 9, -152, 150, -84, 137, -77, 330, -48, 65, 168, -186, 87, -143, 89, -85, -51, 4}
, {-28, -21, -40, -70, 60, -215, 84, 50, -45, -151, -124, -321, -33, 71, -53, 227, -118, 57, 17, 56, -51, 130, -73, 18, -158, 204, 102, 101, 32, -24, 88, 48}
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
  //dense_6_output_type dense_6_output);
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
#include "conv1d_24.c"
#include "weights/conv1d_24.c" // InputLayer is excluded
#include "max_pooling1d_24.c" // InputLayer is excluded
#include "conv1d_25.c"
#include "weights/conv1d_25.c" // InputLayer is excluded
#include "max_pooling1d_25.c" // InputLayer is excluded
#include "conv1d_26.c"
#include "weights/conv1d_26.c" // InputLayer is excluded
#include "max_pooling1d_26.c" // InputLayer is excluded
#include "conv1d_27.c"
#include "weights/conv1d_27.c" // InputLayer is excluded
#include "max_pooling1d_27.c" // InputLayer is excluded
#include "average_pooling1d_6.c" // InputLayer is excluded
#include "flatten_6.c" // InputLayer is excluded
#include "dense_6.c"
#include "weights/dense_6.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_6_output_type dense_6_output) {

  // Output array allocation
  static union {
    conv1d_24_output_type conv1d_24_output;
    conv1d_25_output_type conv1d_25_output;
    conv1d_26_output_type conv1d_26_output;
    conv1d_27_output_type conv1d_27_output;
    average_pooling1d_6_output_type average_pooling1d_6_output;
    flatten_6_output_type flatten_6_output;
  } activations1;

  static union {
    max_pooling1d_24_output_type max_pooling1d_24_output;
    max_pooling1d_25_output_type max_pooling1d_25_output;
    max_pooling1d_26_output_type max_pooling1d_26_output;
    max_pooling1d_27_output_type max_pooling1d_27_output;
  } activations2;


  //static union {
//
//    static input_7_output_type input_7_output;
//
//    static conv1d_24_output_type conv1d_24_output;
//
//    static max_pooling1d_24_output_type max_pooling1d_24_output;
//
//    static conv1d_25_output_type conv1d_25_output;
//
//    static max_pooling1d_25_output_type max_pooling1d_25_output;
//
//    static conv1d_26_output_type conv1d_26_output;
//
//    static max_pooling1d_26_output_type max_pooling1d_26_output;
//
//    static conv1d_27_output_type conv1d_27_output;
//
//    static max_pooling1d_27_output_type max_pooling1d_27_output;
//
//    static average_pooling1d_6_output_type average_pooling1d_6_output;
//
//    static flatten_6_output_type flatten_6_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d_24(
     // First layer uses input passed as model parameter
    input,
    conv1d_24_kernel,
    conv1d_24_bias,
    activations1.conv1d_24_output
  );
 // InputLayer is excluded 
  max_pooling1d_24(
    
    activations1.conv1d_24_output,
    activations2.max_pooling1d_24_output
  );
 // InputLayer is excluded 
  conv1d_25(
    
    activations2.max_pooling1d_24_output,
    conv1d_25_kernel,
    conv1d_25_bias,
    activations1.conv1d_25_output
  );
 // InputLayer is excluded 
  max_pooling1d_25(
    
    activations1.conv1d_25_output,
    activations2.max_pooling1d_25_output
  );
 // InputLayer is excluded 
  conv1d_26(
    
    activations2.max_pooling1d_25_output,
    conv1d_26_kernel,
    conv1d_26_bias,
    activations1.conv1d_26_output
  );
 // InputLayer is excluded 
  max_pooling1d_26(
    
    activations1.conv1d_26_output,
    activations2.max_pooling1d_26_output
  );
 // InputLayer is excluded 
  conv1d_27(
    
    activations2.max_pooling1d_26_output,
    conv1d_27_kernel,
    conv1d_27_bias,
    activations1.conv1d_27_output
  );
 // InputLayer is excluded 
  max_pooling1d_27(
    
    activations1.conv1d_27_output,
    activations2.max_pooling1d_27_output
  );
 // InputLayer is excluded 
  average_pooling1d_6(
    
    activations2.max_pooling1d_27_output,
    activations1.average_pooling1d_6_output
  );
 // InputLayer is excluded 
  flatten_6(
    
    activations1.average_pooling1d_6_output,
    activations1.flatten_6_output
  );
 // InputLayer is excluded 
  dense_6(
    
    activations1.flatten_6_output,
    dense_6_kernel,
    dense_6_bias, // Last layer uses output passed as model parameter
    dense_6_output
  );

}
