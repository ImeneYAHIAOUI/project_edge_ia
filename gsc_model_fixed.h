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

typedef number_t conv1d_20_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_20(
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


const int16_t conv1d_20_bias[CONV_FILTERS] = {-256, -290, -430, -599, -124, -19, -45, -542}
;

const int16_t conv1d_20_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{14, 208, 410, 264, 128, 44, -60, -119, -54, 16, 77, 148, 242, 127, 55, -268, -273, -39, -13, 171, 322, 329, 200, 78, 64, -232, -285, -153, 132, 312}
, {171, -70, 24, 125, 0, -90, -98, -248, -137, 141, 220, 224, 138, 239, 9, -54, -185, -90, -55, -75, -93, 102, 156, 233, 154, -214, -320, -207, -66, 423}
}
, {{198, -3, 460, 287, -174, 189, 240, 90, 550, 389, -272, -229, -155, 84, 651, 269, -404, -72, -55, -154, 259, 351, -120, -122, -228, -339, 30, 150, -171, 221}
, {-77, -72, 287, 324, 99, 107, -285, -447, 22, 122, -4, 61, -167, -332, 46, -2, -35, 167, 195, -216, -39, 72, 209, 435, 146, 70, 121, -13, 25, 418}
}
, {{226, 136, 2, 76, 301, -3, -70, 396, 507, 371, 336, 332, 147, 118, 329, 112, 22, 162, 58, -297, -223, -288, -426, -192, -206, -366, -49, -27, -344, -223}
, {-6, 139, 163, 95, 278, 343, 250, 70, 177, 248, 222, 181, 231, 67, -155, 26, 106, -222, -374, -45, -235, -346, -327, -65, -51, -192, -316, -221, -109, -215}
}
, {{385, 119, -142, -227, 92, 130, 37, 134, -57, -154, -83, 131, 216, 334, 217, -29, -316, -268, -211, -233, -173, 28, 35, -85, 18, 69, 93, 210, 358, 154}
, {638, 227, -5, -137, 46, 96, -43, 27, 65, -167, -66, 123, 68, 124, 66, 70, 45, -95, -88, -131, 3, -4, 36, 110, 180, 250, 414, 365, 189, 87}
}
, {{277, 128, 35, 17, 158, 209, 102, 124, 275, 142, -58, 94, 263, 271, 130, 71, 183, 96, 32, 91, 140, 189, 23, 42, 141, 1, 93, 87, 154, 217}
, {264, 217, 299, 197, 107, 169, 3, 104, 156, 160, 137, 185, 209, 141, 45, 210, 464, 273, 201, 192, 195, 205, 238, 145, 184, 483, 321, 218, 182, 146}
}
, {{207, 61, -187, -100, 103, 120, 29, -147, -114, 85, 86, 110, 94, 234, 174, -22, -113, -178, -55, 266, 409, 298, 73, -273, -344, -109, 228, 429, 263, -140}
, {272, 30, -198, -367, -103, 423, 402, 113, -175, -384, -394, 46, 393, 517, 305, -256, -446, -472, -234, 3, 199, 165, 12, -263, -430, -142, 208, 257, 86, -213}
}
, {{-138, -64, 204, 12, -203, 13, 121, 61, -300, -270, 190, 268, -232, -441, -49, 173, 36, -205, -193, 53, 142, -14, -232, -187, 43, 15, -198, -191, 79, 55}
, {-353, -256, 257, 32, -429, -297, 427, 321, -401, -626, 122, 426, -80, -550, -112, 408, 53, -266, -287, -34, 169, -9, -193, -177, 91, 180, -110, -330, -86, 362}
}
, {{515, 221, -16, 207, 317, 235, 221, 272, 277, 166, 52, -14, -63, 164, 185, 313, 386, 240, 131, 261, 184, 226, 351, 356, 345, 241, 206, 220, 319, 487}
, {254, 27, -185, -319, -306, -141, -30, 206, -22, -78, -54, -256, -284, -68, -165, -90, 112, -203, -455, -226, 2, 59, 126, 81, -121, -34, -270, -346, -144, 54}
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

typedef number_t max_pooling1d_20_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_20(
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

typedef number_t conv1d_21_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_21(
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


const int16_t conv1d_21_bias[CONV_FILTERS] = {275, 737, -54, 174, -843, 25, -71, 28}
;

const int16_t conv1d_21_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-126, -315, 152}
, {9, -196, -280}
, {-350, 40, -260}
, {75, 9, 146}
, {-1014, -902, -841}
, {-592, -407, -29}
, {-706, -234, -434}
, {-277, -179, 127}
}
, {{-802, -478, -420}
, {-222, 41, -220}
, {-42, -369, 506}
, {-116, 158, -270}
, {-678, 76, 67}
, {-971, -951, -653}
, {-85, 184, 191}
, {-299, -25, 8}
}
, {{108, 12, -260}
, {-316, -244, -304}
, {384, 213, 321}
, {-54, -279, -402}
, {-63, -387, -275}
, {13, -55, 121}
, {90, 140, 348}
, {-135, -99, -457}
}
, {{-132, -193, 44}
, {-187, -188, 127}
, {287, 45, 265}
, {-25, 50, -325}
, {202, -55, 71}
, {-307, -216, -267}
, {140, -138, 125}
, {-516, 34, 301}
}
, {{14, 1, 449}
, {-155, -156, -98}
, {-186, 122, 100}
, {-239, -57, -110}
, {-269, 61, 19}
, {-434, 7, 155}
, {-223, 165, 327}
, {-309, -205, 257}
}
, {{80, 96, -83}
, {255, 203, 46}
, {-190, 23, 259}
, {-220, -193, -185}
, {-243, -237, -153}
, {-167, 210, 217}
, {-270, -19, -130}
, {-251, -223, -20}
}
, {{-103, -39, -106}
, {8, -41, -108}
, {-135, -8, -235}
, {-40, -31, -148}
, {-122, 32, -76}
, {36, -104, -7}
, {-40, -113, 12}
, {-32, -31, 1}
}
, {{-315, -3, 9}
, {-93, -282, 85}
, {-47, -247, -417}
, {-188, -211, 127}
, {124, -16, 434}
, {42, 108, 67}
, {295, 111, -81}
, {-179, -9, -92}
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

typedef number_t max_pooling1d_21_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_21(
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

typedef number_t conv1d_22_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_22(
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


const int16_t conv1d_22_bias[CONV_FILTERS] = {-51, -1528, -511, -1149, -136, -247, 88, -942, -1136, 5, -381, 265, 454, -27, 112, 586}
;

const int16_t conv1d_22_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-308, -127, 127}
, {-417, -939, -391}
, {239, 198, 230}
, {-303, -264, -133}
, {-487, -496, -72}
, {-78, -49, -33}
, {149, -17, -103}
, {-52, 115, -41}
}
, {{-433, 130, 495}
, {-923, -164, 128}
, {22, -53, 204}
, {38, -36, 64}
, {176, 66, 180}
, {36, 59, -130}
, {-102, -23, -15}
, {14, -39, 155}
}
, {{81, -38, 116}
, {-309, -82, -105}
, {464, 307, -14}
, {-271, -87, 78}
, {47, -133, -29}
, {53, -31, -180}
, {6, 16, -26}
, {-420, -200, -331}
}
, {{-592, 139, -55}
, {-120, -56, 297}
, {138, -76, 404}
, {-219, -181, -221}
, {-33, -592, -140}
, {121, 60, -122}
, {93, -147, -24}
, {219, 68, -86}
}
, {{388, 34, -160}
, {368, -131, -353}
, {-142, -536, -341}
, {219, -913, -937}
, {26, -41, -175}
, {-433, 200, -462}
, {-20, 85, 52}
, {194, -428, -327}
}
, {{-167, 58, 42}
, {-581, 30, 103}
, {-206, -367, -128}
, {-219, -28, -225}
, {-155, 2, -124}
, {-56, -101, -41}
, {-31, -72, -19}
, {-196, -206, -142}
}
, {{330, -157, -306}
, {438, 304, -1067}
, {48, 108, 317}
, {-366, 14, 67}
, {-71, -499, -476}
, {-52, -90, -7}
, {-116, -15, -82}
, {-313, 28, 27}
}
, {{-33, 147, -251}
, {-416, 429, -281}
, {174, 67, -290}
, {193, 12, 49}
, {-268, -548, -116}
, {38, 290, -45}
, {-84, 108, -115}
, {109, -676, 138}
}
, {{-315, -575, -73}
, {-161, 387, 301}
, {314, 10, 88}
, {-139, -25, 25}
, {-630, -302, 237}
, {-728, -881, -154}
, {-134, 113, 84}
, {-95, 18, -293}
}
, {{356, 154, -620}
, {631, -277, -438}
, {-163, -124, -415}
, {-31, -147, -233}
, {-205, -314, 224}
, {-408, -265, -16}
, {184, -77, 90}
, {-38, 140, 107}
}
, {{-587, -746, -288}
, {190, 412, 59}
, {65, -65, -385}
, {-56, 135, 222}
, {-339, 123, 19}
, {-242, -686, -154}
, {92, -58, -27}
, {61, 130, -31}
}
, {{-276, -451, -920}
, {118, -874, 472}
, {-328, -711, -471}
, {139, -32, 80}
, {111, -369, -641}
, {-173, 168, -450}
, {22, 67, -75}
, {23, -177, 243}
}
, {{88, -498, -332}
, {20, -338, -195}
, {-67, -32, -22}
, {131, -14, 180}
, {-203, -269, -87}
, {-119, -75, 142}
, {-118, -30, 98}
, {-240, -886, -403}
}
, {{-1422, -209, -368}
, {-1070, -469, -121}
, {-175, 267, 208}
, {-25, 116, -2}
, {-163, -175, -294}
, {135, -150, 304}
, {-43, -169, 72}
, {-261, -839, -1067}
}
, {{-87, -342, -196}
, {-842, -786, -1012}
, {-487, 77, 110}
, {129, 81, -115}
, {-94, -446, -95}
, {-397, -41, -33}
, {101, -62, -40}
, {288, -184, 34}
}
, {{31, 78, 201}
, {-535, -249, -8}
, {-448, -288, -366}
, {-1097, -642, -952}
, {-60, -744, -77}
, {94, -95, 221}
, {102, -55, 36}
, {60, 72, 6}
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

typedef number_t max_pooling1d_22_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_22(
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

typedef number_t conv1d_23_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_23(
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


const int16_t conv1d_23_bias[CONV_FILTERS] = {261, -438, -67, 56, -257, -42, 200, 249, -172, 263, -12, 442, -79, -293, -107, 310, 452, -181, 514, 98, 494, 7, 365, -364, -209, 506, -631, 350, 103, 547, 220, -92}
;

const int16_t conv1d_23_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-113, -204, -396}
, {-320, -209, -211}
, {-275, -493, -103}
, {-527, -366, -534}
, {-241, -125, 352}
, {48, -91, -302}
, {-633, -1013, 171}
, {-275, -404, 140}
, {-634, -410, -85}
, {-170, -364, 211}
, {-425, -311, 213}
, {-66, -527, -33}
, {-284, -917, 298}
, {-48, -513, -44}
, {-87, -767, 135}
, {-189, -467, 73}
}
, {{-116, -48, -155}
, {-254, -198, -319}
, {-68, -51, -11}
, {-116, -80, 7}
, {51, -41, -95}
, {43, -104, -55}
, {-83, -272, -138}
, {-200, -403, -160}
, {-122, -155, -336}
, {-31, -206, -234}
, {-155, -450, -185}
, {39, -241, -118}
, {-103, -162, -143}
, {48, -74, -85}
, {-68, -252, -283}
, {-184, -184, -182}
}
, {{-37, -209, 3}
, {-126, -127, -40}
, {-221, -394, -6}
, {-291, -177, -125}
, {204, -92, 33}
, {-39, -115, 20}
, {-388, -114, 217}
, {-201, 101, -187}
, {129, -109, -106}
, {-334, 13, -645}
, {67, 9, 171}
, {11, -112, 108}
, {-262, -86, 431}
, {-458, -308, 77}
, {156, -238, 188}
, {-308, -293, -216}
}
, {{-55, -390, 1}
, {-664, -440, 48}
, {-126, -326, -67}
, {9, -113, -346}
, {54, 28, 233}
, {-106, -45, -108}
, {107, 9, -276}
, {-121, -216, -51}
, {-276, -289, -33}
, {288, 218, 247}
, {-364, -123, -384}
, {-505, -262, 3}
, {56, -286, -85}
, {27, 12, -171}
, {-294, -11, -31}
, {-74, 43, 107}
}
, {{-40, 62, -107}
, {-107, -39, -178}
, {-121, -333, -150}
, {-98, -221, -80}
, {78, -20, -8}
, {-104, 19, -63}
, {-85, -205, 92}
, {-79, -520, -80}
, {-88, -248, -110}
, {-289, -152, -273}
, {-53, -422, 71}
, {-378, 71, 261}
, {-301, -75, 123}
, {-97, -399, -555}
, {-190, -226, -253}
, {-213, 142, -583}
}
, {{81, -423, -315}
, {-421, -581, -454}
, {-511, 56, 14}
, {-122, -238, -486}
, {-658, -52, 255}
, {-8, 57, -89}
, {249, 110, 325}
, {-61, -475, -776}
, {-236, 68, -423}
, {-510, -271, 256}
, {-193, -270, -248}
, {-52, -625, -238}
, {-152, 21, 188}
, {-438, 30, 65}
, {-246, -724, -813}
, {2, -802, -246}
}
, {{-107, 221, 88}
, {-465, -21, -876}
, {-68, 25, -498}
, {45, -215, -115}
, {129, 18, -149}
, {130, -115, -202}
, {266, 51, 114}
, {7, 32, -485}
, {-311, 170, -522}
, {240, -376, -146}
, {-621, 3, -176}
, {-498, -247, -470}
, {-271, -431, -552}
, {-313, -495, -373}
, {-11, 85, 88}
, {393, 69, -23}
}
, {{-356, 56, 157}
, {-168, -215, -104}
, {-425, -727, 369}
, {-373, -170, -204}
, {-219, -68, -208}
, {127, -68, 47}
, {-656, 84, 163}
, {-137, -274, -30}
, {-74, 138, -133}
, {-109, -401, -124}
, {-123, -226, -196}
, {-59, -505, -65}
, {-227, -92, 153}
, {-288, -476, 256}
, {-85, -607, 173}
, {100, -415, 25}
}
, {{-207, -219, -87}
, {-29, -220, -27}
, {-168, -198, -67}
, {-167, -232, 16}
, {61, -8, -35}
, {76, -37, -167}
, {-84, -11, -229}
, {-73, -28, -156}
, {-145, -195, -54}
, {-305, -266, -128}
, {-120, -271, -234}
, {-121, 2, -164}
, {-48, -103, -139}
, {-8, -146, 52}
, {21, -202, -124}
, {-155, -206, -189}
}
, {{-11, 178, -290}
, {128, -6, 168}
, {173, -105, -230}
, {-107, -35, -439}
, {16, 0, 102}
, {264, 70, 195}
, {-341, 68, -388}
, {-218, -262, 131}
, {34, -237, -490}
, {37, -342, -138}
, {35, -398, -386}
, {-14, -391, -1023}
, {39, -171, -61}
, {-487, 116, 171}
, {-176, -264, -500}
, {-216, 134, 183}
}
, {{257, 252, 141}
, {-309, 194, -428}
, {-198, 44, -223}
, {-485, -155, -219}
, {-17, 194, 402}
, {19, -107, -130}
, {222, -280, 18}
, {-626, -175, -571}
, {-150, -425, 29}
, {-220, -88, -29}
, {-270, -334, -275}
, {-614, 163, -117}
, {-250, -122, -76}
, {-193, -157, 71}
, {-148, -201, -18}
, {277, 193, -29}
}
, {{310, -236, -263}
, {-268, -163, -165}
, {-657, -480, 195}
, {261, 5, -227}
, {-118, 388, 100}
, {-67, 150, 117}
, {-115, 362, 255}
, {-130, -146, 152}
, {-149, -170, -197}
, {519, 42, -115}
, {-1, 150, -145}
, {-167, 76, -180}
, {-214, -293, 150}
, {-631, -317, 118}
, {-120, -171, -197}
, {-460, -886, -1190}
}
, {{-125, -30, -52}
, {-77, -72, -5}
, {-209, -51, -155}
, {-101, 35, -146}
, {62, 31, -25}
, {-36, -61, 1}
, {-290, 28, -310}
, {-60, -41, -119}
, {-92, -8, -77}
, {-52, -99, -151}
, {45, 51, -129}
, {-149, -188, -296}
, {-142, -38, -157}
, {54, -31, -30}
, {-102, -108, -103}
, {-127, 11, -62}
}
, {{-349, 447, 6}
, {-357, -521, -358}
, {-106, -558, -280}
, {-32, -57, -246}
, {-42, -61, -47}
, {-45, -4, 72}
, {-897, 252, 285}
, {-210, -176, -210}
, {185, -250, -323}
, {-694, 479, 176}
, {-55, 34, 24}
, {-23, -259, -200}
, {-22, 49, -480}
, {10, -45, -427}
, {80, -8, -337}
, {-79, -95, -136}
}
, {{-138, 38, -168}
, {-134, -158, -99}
, {-120, -127, -295}
, {-191, -246, -60}
, {-131, -201, -3}
, {-19, -131, 77}
, {-7, -162, -380}
, {-220, -281, 45}
, {-230, -293, -78}
, {-437, -218, -55}
, {-120, -287, -211}
, {98, -360, -351}
, {-283, -72, -202}
, {-232, -211, -254}
, {-111, -92, -84}
, {-196, -92, -129}
}
, {{-103, -24, -20}
, {-222, -331, -244}
, {-253, -505, 249}
, {-243, -116, -351}
, {-101, 125, 94}
, {75, 310, 84}
, {-455, 266, -405}
, {-90, -368, -548}
, {-219, -688, -465}
, {117, 27, -129}
, {-644, 130, -5}
, {77, 110, 256}
, {-186, 9, 77}
, {264, -230, 24}
, {-335, 192, 132}
, {399, -58, 310}
}
, {{127, 229, 147}
, {-527, -626, -483}
, {-110, -213, -346}
, {-369, -371, 168}
, {-673, -875, -640}
, {-169, 338, 208}
, {319, 197, -6}
, {-334, -336, -313}
, {-522, -722, -317}
, {79, 278, -426}
, {-363, -168, -47}
, {-1038, 0, -379}
, {396, -179, -9}
, {158, -55, -38}
, {-292, -103, 19}
, {-232, 157, -266}
}
, {{-104, 37, -242}
, {-220, -130, -78}
, {-1, -32, -149}
, {-13, -130, -100}
, {-55, 5, -90}
, {-30, 7, -4}
, {-136, -77, -45}
, {38, 3, -99}
, {23, -80, -196}
, {-169, -121, -107}
, {-35, -210, -73}
, {-59, -28, -67}
, {-33, -8, -94}
, {-47, -3, 36}
, {-37, -168, -112}
, {-188, -46, -30}
}
, {{129, 40, 54}
, {-45, -18, -189}
, {198, 59, -135}
, {186, -218, -623}
, {-481, -346, -26}
, {63, -8, -210}
, {19, 181, 43}
, {107, -229, -259}
, {-131, -170, -308}
, {200, -2, -186}
, {-339, -530, -780}
, {99, 100, -25}
, {48, 117, 266}
, {232, 33, 84}
, {-576, -223, -166}
, {118, -166, 352}
}
, {{-166, -189, -210}
, {-39, -97, -107}
, {18, -188, -81}
, {-153, -128, -97}
, {-378, -132, -55}
, {118, -181, -101}
, {58, 50, -92}
, {-189, 15, -185}
, {-148, -177, -101}
, {-333, 289, -270}
, {113, -230, -213}
, {253, -16, -72}
, {-198, -135, -194}
, {-40, 7, -72}
, {18, -13, -78}
, {-462, -129, 16}
}
, {{-407, 85, 74}
, {-340, 118, -67}
, {-53, -134, 105}
, {-540, -417, -116}
, {569, 339, 286}
, {22, 157, -96}
, {155, -291, -162}
, {-974, -413, -150}
, {197, 146, 91}
, {221, 108, 55}
, {168, 70, -85}
, {366, 122, 137}
, {-633, -112, -412}
, {-370, -522, -295}
, {-314, 156, 32}
, {-732, -407, -120}
}
, {{-159, -38, 55}
, {-69, -207, -64}
, {-78, -34, -64}
, {-212, -260, -59}
, {4, -224, -213}
, {-67, -84, -379}
, {-389, 399, 169}
, {-144, -111, -49}
, {-91, -309, -265}
, {114, -216, -77}
, {-156, -448, -313}
, {18, -259, -287}
, {-254, -7, 9}
, {-179, 224, 120}
, {-9, -467, -443}
, {-35, -111, -143}
}
, {{-94, 48, -13}
, {-251, -403, -300}
, {-25, -73, -402}
, {246, 46, 43}
, {-97, -371, -102}
, {-32, -111, 39}
, {39, 131, 199}
, {-861, -376, 227}
, {-33, -381, -55}
, {140, 324, 195}
, {-172, -101, -170}
, {-369, 88, -54}
, {205, 135, -252}
, {-112, -420, -597}
, {154, -368, -155}
, {33, 69, -547}
}
, {{23, -153, -154}
, {-121, -131, -89}
, {-92, -39, -521}
, {-97, -302, -111}
, {-112, -124, 11}
, {-33, -88, -119}
, {-187, 47, -157}
, {68, -190, -144}
, {-248, -51, -96}
, {-345, -495, -150}
, {-139, -31, -91}
, {-141, -373, -450}
, {-354, -384, 39}
, {-30, -238, -126}
, {-139, -128, -33}
, {-57, -165, -215}
}
, {{-117, -172, -132}
, {-21, -150, -53}
, {34, 1, -7}
, {-26, -82, -35}
, {-68, 8, -26}
, {-28, -126, -54}
, {-77, -96, -169}
, {-203, -129, 45}
, {-106, -84, -57}
, {-7, -52, -16}
, {-222, -188, -29}
, {-255, -263, -258}
, {-103, -162, -206}
, {-21, -30, -175}
, {-192, -180, -230}
, {-17, -111, -84}
}
, {{-105, -134, -402}
, {-564, -335, -277}
, {-136, -388, -266}
, {-19, -207, -244}
, {346, -176, 111}
, {-6, 61, 52}
, {-12, -107, -468}
, {-252, -193, -165}
, {-277, -194, -234}
, {20, 16, -88}
, {22, -236, -208}
, {0, -275, -128}
, {-53, -142, -229}
, {-1, -361, -171}
, {323, 43, 49}
, {-243, 41, -4}
}
, {{-101, -188, -107}
, {15, -28, 188}
, {-195, -87, 37}
, {-254, -278, -60}
, {-135, -385, -39}
, {127, 40, -22}
, {-619, -36, -233}
, {-336, 55, -73}
, {181, -286, -238}
, {-44, -296, -320}
, {-3, -32, 143}
, {108, 166, 134}
, {-87, 127, 348}
, {-438, 371, -27}
, {-271, 244, -277}
, {-217, -267, -447}
}
, {{-5, -46, 86}
, {68, 51, 211}
, {-34, 4, -16}
, {-189, -273, -388}
, {-197, -197, -2}
, {-16, -161, -215}
, {529, -531, -9}
, {-21, -195, -15}
, {-145, -46, -502}
, {-21, 126, -38}
, {-64, 15, -151}
, {-258, -210, -383}
, {437, 60, -10}
, {371, -294, -71}
, {-399, 67, -164}
, {-254, 51, -654}
}
, {{82, 190, -229}
, {-68, -298, -689}
, {188, -72, -155}
, {-104, -17, 14}
, {-101, 272, 501}
, {6, -199, -156}
, {-101, -80, -437}
, {262, -66, -92}
, {-74, -184, -321}
, {-427, -12, -176}
, {251, -169, -211}
, {202, 72, 0}
, {469, -70, -216}
, {226, -13, -510}
, {243, 46, -133}
, {-1117, 36, 194}
}
, {{-116, -264, -140}
, {-489, -300, -235}
, {-45, 44, -256}
, {-18, -120, -171}
, {49, 98, 204}
, {-39, 34, 93}
, {-433, -193, -104}
, {96, -48, 215}
, {215, -283, -365}
, {-49, 246, -146}
, {33, -129, -506}
, {-7, 143, -607}
, {35, -1, 25}
, {90, 202, -152}
, {215, -17, -50}
, {275, -119, 320}
}
, {{-793, 249, 102}
, {-770, -386, 17}
, {-585, -137, 139}
, {-249, -84, 117}
, {-690, 1006, 340}
, {-28, -20, -208}
, {288, 95, 135}
, {-987, -206, -51}
, {-548, -328, 53}
, {-260, 288, -136}
, {-581, -332, -62}
, {-377, -505, -39}
, {-721, -798, 249}
, {-537, -242, -80}
, {-294, -389, 310}
, {-1568, -219, -214}
}
, {{-253, -175, -224}
, {-38, -112, -101}
, {-178, -193, -192}
, {-129, -255, -98}
, {-101, -45, -149}
, {-4, -92, -89}
, {-57, -102, -177}
, {-150, -218, -47}
, {-172, -139, -156}
, {-109, 6, -129}
, {-75, -216, -77}
, {-135, -91, -122}
, {-127, -133, -122}
, {43, -20, -14}
, {-157, -165, -161}
, {-106, -127, -33}
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

typedef number_t max_pooling1d_23_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_23(
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

typedef number_t average_pooling1d_5_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d_5(
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

//typedef number_t *flatten_5_output_type;
typedef number_t flatten_5_output_type[OUTPUT_DIM];

#define flatten_5 //noop (IN, OUT)  OUT = (number_t*)IN

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
#define FC_UNITS 4
#define ACTIVATION_LINEAR

typedef number_t dense_5_output_type[FC_UNITS];

static inline void dense_5(
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
#define FC_UNITS 4


const int16_t dense_5_bias[FC_UNITS] = {-60, 29, -16, 44}
;

const int16_t dense_5_kernel[FC_UNITS][INPUT_SAMPLES] = {{80, -308, 24, 36, -252, -351, 196, -263, -117, -77, 17, -134, -97, 1, 0, -193, 248, 89, -135, 151, 120, -202, 158, -129, 124, 96, -27, 62, -101, 69, 356, -189}
, {519, -89, 66, -323, 246, 26, -186, 84, -115, 25, -358, 159, 77, -221, -152, 42, -467, 86, -179, 112, -10, 61, -28, 139, -80, 102, 98, -55, 94, 22, 317, -21}
, {162, -240, -49, 72, 32, 345, 207, 64, -18, 55, -8, -24, 51, 126, 4, 244, 46, -53, 164, -56, -97, 172, -105, 192, 149, 55, -20, -55, 7, 147, -269, -29}
, {-316, 8, -73, 43, 65, 247, 85, 20, -128, 127, 91, 163, -45, 148, -156, -146, 106, -27, 142, -2, -34, 193, 119, 70, 103, -288, 32, 120, -34, -45, -197, -38}
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

#define MODEL_OUTPUT_SAMPLES 4
#define MODEL_INPUT_SAMPLES 10000 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 2

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_5_output_type dense_5_output);
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
