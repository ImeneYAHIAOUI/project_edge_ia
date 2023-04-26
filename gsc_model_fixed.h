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

typedef number_t conv1d_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d(
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


const int16_t conv1d_bias[CONV_FILTERS] = {-1004, -1169, -866, -1410, -2321, 97, -157, -41}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{717, 466, 180, 221, 277, 226, 308, 466, 129, -109, -18, 153, 588, 568, 371, -67, -151, -18, 259, 465, 224, 124, 308, 469, 409, 427, 517, 650, 596, 205}
, {-112, -617, -213, 30, -208, -193, 254, 166, -81, -402, -822, -607, 47, 117, -202, 13, -32, -287, -469, -369, -242, -106, -144, -300, -249, -274, 68, 165, -102, -449}
}
, {{-708, -580, -360, -17, 88, 110, 169, 391, 524, 352, 145, -178, -398, -491, -373, 121, 506, 528, 619, 445, 402, 92, -113, -371, -568, -618, -372, 196, 486, 572}
, {138, 340, 438, 377, 432, 259, -7, -27, -153, -229, 4, 157, 71, 348, 311, 395, 345, 158, -26, -64, 3, -79, 187, 69, -111, -227, -142, -184, 138, 380}
}
, {{-745, -671, -670, -835, -807, -647, -488, -466, -286, -9, -68, -81, -112, -480, -447, -566, -743, -471, 154, 173, 145, 159, -175, -386, -97, -193, -398, -490, -700, -686}
, {-513, -35, 82, 89, 180, -35, 129, 309, 323, 255, -32, -136, -124, -119, -198, -406, -665, -377, 184, 61, -38, -73, -496, -541, 111, 95, -276, -474, -380, -632}
}
, {{727, 12, 8, 431, 381, 98, -102, 50, 581, 453, 117, 216, -52, -407, -549, -611, -598, -483, -570, -792, -618, -542, -473, -496, -510, -238, 161, 184, 151, 751}
, {452, -93, -131, 104, 477, 569, 408, 561, 680, 220, -95, 164, 54, -126, -155, -360, -540, -644, -776, -716, -654, -302, -38, -29, -64, -52, -47, -205, 24, 640}
}
, {{748, 517, 120, -155, -343, -506, -335, 353, 614, 592, 594, 174, -614, -826, -618, -173, 157, 86, -162, -26, 147, -155, -199, -212, -276, -233, -105, -277, -6, 272}
, {597, 520, 136, -98, 14, 68, 16, 38, -52, 118, 209, 253, -21, -133, -309, -226, -60, -251, -94, 182, 229, 141, 44, -213, -412, -140, -80, -205, 58, 90}
}
, {{636, 445, 455, 672, 705, 243, 115, 441, 449, 525, 699, 291, 140, 361, 151, 41, 309, 160, 278, 397, 499, 701, 925, 470, 575, 640, 511, 455, 760, 1162}
, {348, -106, 165, 60, -147, 389, 379, 213, 431, 187, -103, -140, -162, -98, 216, 245, -32, -26, 143, 67, 379, 738, 570, 385, 793, 749, 591, 452, 426, 655}
}
, {{2, 345, 73, -3, -49, -319, -266, -283, -530, -562, -275, -189, -82, 336, 242, -95, 135, 125, -257, -475, -525, -532, -247, 65, 1, -220, -326, -107, -451, -795}
, {-342, -201, -341, -363, -158, -327, -317, 216, 254, 287, 319, 110, 73, 145, -155, -119, 352, 473, 485, 724, 490, 602, 889, 838, 842, 854, 725, 945, 657, 4}
}
, {{-996, -903, 119, -69, -905, -304, 439, 127, -232, -582, -646, 0, 646, 385, 522, 654, 559, 117, 66, 517, 1077, 470, -779, -592, 700, 698, 75, 206, -27, -667}
, {-584, -498, 407, 111, -654, -524, -145, 237, 598, 215, -181, 50, 88, -211, 12, 95, -149, -325, -330, 351, 679, -150, -931, -425, 742, 591, 219, 274, -70, -1022}
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

typedef number_t max_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d(
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

typedef number_t conv1d_1_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_1(
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


const int16_t conv1d_1_bias[CONV_FILTERS] = {771, -70, -116, -254, -3, 863, -21, 551}
;

const int16_t conv1d_1_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-414, -84, -169}
, {-358, -356, -417}
, {365, -198, -1030}
, {-624, -501, -32}
, {-238, -237, -375}
, {-356, -66, -54}
, {-229, -143, -1282}
, {39, -179, -182}
}
, {{25, 18, -227}
, {-22, -90, -106}
, {-103, 44, -108}
, {-206, -198, -73}
, {-154, -186, -2}
, {-143, -227, -60}
, {72, -108, -100}
, {-42, -41, -93}
}
, {{346, 402, -231}
, {252, 181, -106}
, {568, -86, 160}
, {-181, -192, 18}
, {-10, -300, -456}
, {-163, 345, 372}
, {-362, 22, 192}
, {-510, -397, -260}
}
, {{73, -113, -171}
, {-54, -197, -491}
, {-424, -501, -161}
, {-90, -11, -111}
, {-436, -659, -615}
, {-149, -269, -221}
, {-249, -349, -314}
, {-195, -113, -180}
}
, {{-234, 215, -119}
, {463, 456, -42}
, {-25, 370, 426}
, {-440, -268, -541}
, {-177, 126, 35}
, {-393, 336, -608}
, {40, 41, -157}
, {-433, -386, -659}
}
, {{-773, -716, -330}
, {-854, -170, 186}
, {244, 93, 449}
, {-1089, -1108, -215}
, {-484, -747, -112}
, {-859, -1067, -1998}
, {-1472, -1146, -931}
, {-487, -701, -630}
}
, {{95, 10, -133}
, {-315, 100, -219}
, {-490, -475, -281}
, {-340, -97, -43}
, {-23, 45, -25}
, {-421, -578, -225}
, {-249, -415, -180}
, {-339, -181, -453}
}
, {{-80, 501, 75}
, {-574, 184, -340}
, {1037, -701, -169}
, {311, -586, -959}
, {-488, -1168, -81}
, {-956, 490, 181}
, {-500, -336, 154}
, {93, -347, -114}
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

typedef number_t max_pooling1d_1_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_1(
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

typedef number_t conv1d_2_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_2(
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


const int16_t conv1d_2_bias[CONV_FILTERS] = {-118, -110, -138, -269, -363, 1278, -270, -99, 724, -40, -303, 1092, -204, -920, -45, 195}
;

const int16_t conv1d_2_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-112, -123, -183}
, {-64, -124, 30}
, {-106, -283, -145}
, {-138, 64, 316}
, {-319, -59, -132}
, {21, 72, -455}
, {-146, -161, -130}
, {-469, -273, -52}
}
, {{-136, -139, -70}
, {-49, -41, 37}
, {-5, -199, -123}
, {-135, -63, -28}
, {-89, -202, -13}
, {12, -46, 3}
, {-21, -8, -107}
, {-286, -51, 0}
}
, {{-149, -217, 156}
, {49, 94, 36}
, {-29, -135, -85}
, {-60, -61, -75}
, {-197, -392, -361}
, {35, -91, -284}
, {-78, -83, -24}
, {-329, -129, -468}
}
, {{-193, -13, 26}
, {44, -7, 101}
, {-215, -256, -252}
, {-37, -56, -141}
, {-202, 0, -247}
, {-166, -196, 16}
, {-79, -25, -1}
, {-240, -44, -82}
}
, {{-6, -22, -104}
, {-67, -149, -103}
, {-87, -141, -302}
, {-154, 259, 104}
, {-5, -158, -89}
, {-134, 51, -7}
, {-100, 4, 7}
, {-156, -130, -214}
}
, {{-299, -464, 209}
, {20, 75, 133}
, {-281, -561, -812}
, {105, 161, 227}
, {-897, -40, -150}
, {469, 348, -704}
, {-14, -22, -14}
, {140, 225, -169}
}
, {{61, -30, -171}
, {133, -120, 84}
, {-94, -237, -111}
, {190, 183, -154}
, {-204, -143, -86}
, {-118, -2, -37}
, {-81, -143, 53}
, {-83, -155, -132}
}
, {{-39, 217, 0}
, {-60, 65, 48}
, {-87, -1041, -480}
, {159, 588, 207}
, {-813, -612, -33}
, {258, -338, 512}
, {2, 105, -58}
, {-179, -91, -46}
}
, {{767, 536, -756}
, {25, -180, -61}
, {-397, -1206, -834}
, {193, 629, 505}
, {-531, 7, -157}
, {-216, 215, -411}
, {93, 123, 239}
, {-492, -916, -1400}
}
, {{-572, -135, -5}
, {-150, 160, 68}
, {-67, -189, -243}
, {-20, -59, -67}
, {-234, -553, -359}
, {-112, -294, -73}
, {-49, -32, -37}
, {-550, -407, -239}
}
, {{819, 894, -813}
, {-23, 14, 4}
, {-30, 30, 93}
, {-83, -275, -58}
, {-71, -579, -776}
, {470, -687, -122}
, {-64, -255, -205}
, {-1231, -200, -410}
}
, {{-907, 95, -1159}
, {-11, -59, 60}
, {-101, 56, -213}
, {141, 185, 139}
, {-139, -721, -1345}
, {295, -214, -427}
, {-157, 137, -97}
, {-1690, -681, -845}
}
, {{-179, -329, -295}
, {-75, 67, 118}
, {-331, -507, -208}
, {-373, 242, -161}
, {-272, -101, -9}
, {-11, -186, -131}
, {-188, -89, -102}
, {-11, -50, -134}
}
, {{235, -1254, -435}
, {1, 86, 8}
, {104, -403, -575}
, {430, 627, 295}
, {-85, 182, 12}
, {70, 765, 782}
, {-480, -214, -106}
, {-14, 143, 71}
}
, {{-200, -148, -37}
, {95, -43, 121}
, {-103, -84, -14}
, {-122, -185, -75}
, {-212, -41, -193}
, {6, 29, -56}
, {-75, -23, -167}
, {-305, -244, -176}
}
, {{-1143, -522, 1165}
, {-102, -70, 76}
, {-82, -172, 92}
, {-72, 564, -2}
, {-54, -80, 127}
, {355, -1123, -1356}
, {-333, 298, 99}
, {-525, -1010, -774}
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

typedef number_t max_pooling1d_2_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_2(
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

typedef number_t conv1d_3_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_3(
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


const int16_t conv1d_3_bias[CONV_FILTERS] = {-1289, -495, 6, -276, -5, -437, -609, -411, -1262, 179, 947, 1255, -390, -726, -216, -47, -22, -800, -269, -436, -283, -165, 1376, -760, -262, -768, -103, -740, -617, -339, -706, -579}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{188, -95, -54}
, {92, -17, 33}
, {-199, 10, 41}
, {12, 52, -30}
, {46, -15, -109}
, {-470, -411, 20}
, {29, -111, -104}
, {462, 177, 131}
, {-203, -11, 270}
, {-145, -199, 22}
, {-69, -12, -40}
, {-276, 379, 423}
, {-33, -74, -58}
, {-492, -115, -105}
, {8, 90, -155}
, {-240, -149, -91}
}
, {{80, 42, -92}
, {67, 27, -64}
, {134, -19, 100}
, {301, 88, -429}
, {54, -41, -27}
, {-398, -272, -565}
, {-130, -36, -281}
, {109, 230, 231}
, {-211, 117, -253}
, {154, -152, -84}
, {-109, -215, -165}
, {2, -85, -105}
, {-79, 66, -134}
, {-397, -269, -382}
, {77, -21, 156}
, {-174, -288, -320}
}
, {{18, -111, -146}
, {62, 81, 98}
, {-45, -115, -151}
, {82, -115, -146}
, {16, -24, -19}
, {-870, 568, 436}
, {-161, -37, -79}
, {-22, 217, 477}
, {322, -1504, 270}
, {-367, -342, -220}
, {-191, -314, -445}
, {-330, -715, 584}
, {-1, -143, -119}
, {-508, -479, -206}
, {-22, -25, -83}
, {-1165, -69, -392}
}
, {{-72, -153, -5}
, {-68, -36, -19}
, {-23, -6, 10}
, {-67, -243, -172}
, {-33, -8, 109}
, {-90, -5, -157}
, {70, -64, -129}
, {-150, 11, -20}
, {-168, -96, -154}
, {-56, -2, -152}
, {-166, -74, -343}
, {-219, 18, -142}
, {-43, -38, -4}
, {-28, -183, -181}
, {27, 57, -142}
, {-307, -112, -142}
}
, {{-36, 282, -63}
, {-45, -5, 23}
, {101, 503, 337}
, {14, -130, -143}
, {33, 30, -144}
, {233, 1273, -3081}
, {-105, 222, 209}
, {251, 252, -596}
, {-1403, -2720, -1058}
, {-155, -123, 89}
, {-216, -228, -1664}
, {256, -41, -1548}
, {-56, 183, 119}
, {-500, -102, -97}
, {55, -135, -117}
, {-112, 12, -53}
}
, {{112, -64, 392}
, {-48, -146, -112}
, {-113, -147, -40}
, {-73, -102, -15}
, {-45, -4, -7}
, {377, -489, 166}
, {-23, -129, -55}
, {247, -87, -130}
, {428, -638, -404}
, {-96, -61, 21}
, {-417, -156, -111}
, {388, 136, 52}
, {-99, -45, 44}
, {-297, -120, -98}
, {-79, 21, -4}
, {-204, -713, -167}
}
, {{-94, -204, 54}
, {-11, -104, -60}
, {-33, 226, 112}
, {266, 217, -53}
, {-65, 87, 24}
, {-145, -900, 408}
, {-170, -229, 6}
, {248, -158, 83}
, {82, -298, 645}
, {-9, 14, -60}
, {-322, 0, -246}
, {-764, -43, 241}
, {-21, 137, 26}
, {-235, -120, -211}
, {-152, -34, -176}
, {-393, -881, -343}
}
, {{137, 296, 155}
, {90, -61, -60}
, {-61, -112, -184}
, {-226, 12, 45}
, {10, -74, 196}
, {-86, -113, -153}
, {100, 177, -220}
, {35, 258, -392}
, {-387, -50, -304}
, {-83, -166, -25}
, {-84, -418, -351}
, {-382, -232, -151}
, {-12, -155, -7}
, {-117, -360, -279}
, {-68, 117, -73}
, {-29, -324, -289}
}
, {{-38, 27, 40}
, {-66, 60, -54}
, {-369, -488, -47}
, {-12, -70, -89}
, {62, -101, -10}
, {71, 269, 376}
, {-13, 46, 294}
, {9, 334, -507}
, {101, 271, -771}
, {-464, -172, -183}
, {-822, -678, -778}
, {-163, 468, -327}
, {37, -117, -105}
, {-201, -290, -189}
, {-26, 38, 16}
, {-18, -229, -628}
}
, {{-1, 317, 105}
, {178, -124, -65}
, {192, -192, -157}
, {-182, -207, -9}
, {-109, 100, 173}
, {-392, 157, 54}
, {43, 4, 422}
, {-339, 12, -604}
, {75, -688, -722}
, {138, -311, -475}
, {-539, -289, -564}
, {370, 570, -1053}
, {60, 75, 43}
, {-710, -449, -595}
, {-84, 21, 161}
, {-349, -673, -425}
}
, {{2, -152, -189}
, {-107, -89, -4}
, {-57, -59, -59}
, {-119, -104, -7}
, {-290, -300, -131}
, {780, 608, -1585}
, {-41, -190, 63}
, {429, 191, -257}
, {-1063, -1050, 347}
, {-29, 133, -242}
, {147, -41, 71}
, {-777, -263, -53}
, {-46, -75, 80}
, {-72, -21, -468}
, {29, 41, 0}
, {-50, -141, 2}
}
, {{-323, -197, -194}
, {51, 82, -23}
, {-27, 26, 37}
, {77, 25, 43}
, {89, -219, 30}
, {-369, -1494, -542}
, {227, -127, 39}
, {393, -1114, -554}
, {-504, -725, -288}
, {93, -174, -5}
, {-61, 20, -168}
, {-192, 29, 86}
, {92, -178, 23}
, {-75, -9, -89}
, {-21, -65, 104}
, {48, 14, 144}
}
, {{201, 35, -127}
, {2, -150, 20}
, {-117, -153, -145}
, {-103, -50, 129}
, {16, 30, -89}
, {31, -345, -40}
, {-66, -92, -1}
, {170, -268, -58}
, {-175, 226, 403}
, {29, -144, -270}
, {-164, -265, -738}
, {-419, 107, -335}
, {-78, -77, 9}
, {-502, -233, -306}
, {120, -1, -94}
, {-301, -135, -259}
}
, {{-83, -7, 0}
, {-76, 102, -103}
, {-20, -23, -317}
, {-152, 60, 18}
, {-135, 49, -92}
, {-446, -338, -356}
, {-126, 18, -122}
, {8, 21, -9}
, {-282, -118, -285}
, {-87, -59, -333}
, {-477, -127, -449}
, {36, -109, -404}
, {-31, -149, -30}
, {-299, -217, -156}
, {-100, 84, -15}
, {-242, -132, -143}
}
, {{139, 103, 110}
, {28, -104, 53}
, {202, -90, -256}
, {307, -48, -13}
, {-98, 111, 210}
, {-150, -289, -217}
, {-36, -14, -158}
, {106, 66, -60}
, {-317, -76, -89}
, {54, -82, 262}
, {-198, -209, -229}
, {-177, -141, 3}
, {145, 256, 231}
, {-671, -301, -76}
, {49, -86, -85}
, {-69, -106, -233}
}
, {{32, -37, -73}
, {-47, -79, -54}
, {-110, -79, -111}
, {-66, 16, -85}
, {31, -137, -12}
, {-121, -141, -33}
, {92, -67, -66}
, {-93, -156, -123}
, {-148, -82, 11}
, {-37, -142, -95}
, {-4, -64, -88}
, {-7, -88, -75}
, {-30, 2, 38}
, {-48, -19, -51}
, {-92, -33, -156}
, {-92, -127, -100}
}
, {{-41, -144, 73}
, {7, 83, 32}
, {-33, 54, -102}
, {-110, -6, -146}
, {-204, -165, 32}
, {-263, 4, -550}
, {159, -60, 128}
, {-45, -583, -60}
, {-311, -518, -219}
, {-228, -428, 129}
, {-526, -280, -44}
, {-437, -443, -10}
, {35, 70, 283}
, {-234, -714, -465}
, {-86, -19, 122}
, {-207, -98, -155}
}
, {{-152, 12, -30}
, {15, -39, -6}
, {65, 218, -112}
, {4, -73, 146}
, {-89, 35, 18}
, {-326, -313, -220}
, {133, -133, 166}
, {281, -3, 330}
, {-341, -7, -199}
, {-95, -67, -208}
, {-492, -790, -244}
, {-310, 359, -141}
, {51, 63, 73}
, {-310, -630, -266}
, {-69, -123, -137}
, {-510, -719, -116}
}
, {{-107, -53, 66}
, {16, -5, -6}
, {-12, -69, -24}
, {26, 29, 12}
, {-18, -9, -68}
, {-250, -134, 24}
, {-37, -104, 19}
, {-26, -75, -107}
, {56, -88, -136}
, {-10, -47, 14}
, {-97, -52, -122}
, {-37, -133, -116}
, {-15, -10, 58}
, {-296, -313, -129}
, {-64, -117, -7}
, {-360, -231, -38}
}
, {{-38, -127, 87}
, {-7, 51, -20}
, {-31, -135, 43}
, {49, 26, -214}
, {-6, -31, 283}
, {-213, 4, -247}
, {113, -5, -46}
, {-124, -94, 99}
, {-110, 13, -128}
, {-108, -132, -6}
, {-114, -188, -236}
, {-3, -17, -4}
, {5, -208, 66}
, {-52, -156, -155}
, {51, 31, -48}
, {-416, -349, -65}
}
, {{-41, 29, 62}
, {64, 73, -131}
, {-52, -245, -192}
, {-146, -75, -25}
, {-15, -112, -74}
, {-182, -117, -39}
, {-78, -15, -238}
, {-117, -85, -37}
, {-60, -8, 21}
, {-154, -154, 2}
, {-250, -49, -5}
, {-41, -25, -152}
, {-17, 234, -82}
, {-128, -165, -246}
, {-115, -5, -16}
, {-344, -172, -104}
}
, {{-47, -94, 139}
, {-39, 61, 0}
, {-270, -84, -94}
, {227, -151, -27}
, {-51, -94, -208}
, {521, 62, 129}
, {-139, 122, 86}
, {133, -71, -92}
, {-89, -125, -16}
, {-556, -131, -308}
, {-224, -257, -1315}
, {-429, 338, 150}
, {148, 93, 355}
, {-901, -146, -180}
, {-158, 21, -84}
, {-279, -24, 49}
}
, {{111, 312, 209}
, {6, -11, -197}
, {158, -257, -86}
, {-127, -171, -106}
, {92, -133, 206}
, {-661, 902, -495}
, {-285, -175, 32}
, {-375, -156, 252}
, {171, -524, -402}
, {217, -286, -174}
, {-17, -41, 43}
, {293, -179, -286}
, {58, -151, -84}
, {-241, -53, -385}
, {97, -100, -41}
, {-62, 64, -180}
}
, {{-92, 268, 47}
, {12, -49, 122}
, {80, -24, -79}
, {-141, 434, -99}
, {-285, -182, -92}
, {-330, -299, -716}
, {153, -113, 48}
, {48, -140, -87}
, {161, -715, 470}
, {-31, -337, -148}
, {-106, -659, -277}
, {175, -521, -473}
, {60, 74, 151}
, {-157, -527, -299}
, {-9, 23, 248}
, {-209, -755, -205}
}
, {{185, -110, -16}
, {47, 18, -58}
, {28, -58, -237}
, {-145, 59, -93}
, {-73, 5, -45}
, {-18, 12, -66}
, {-112, -36, -71}
, {-106, -151, 324}
, {24, -90, -69}
, {-174, -414, -61}
, {-195, -284, -143}
, {-153, -167, -26}
, {-25, -132, 195}
, {-92, -171, -310}
, {-107, -148, -122}
, {-1, -261, -133}
}
, {{56, -113, -86}
, {-106, -13, 9}
, {-16, 280, -72}
, {-80, 28, -46}
, {30, 64, -12}
, {-274, -204, -45}
, {-103, -57, -7}
, {64, -121, -125}
, {-22, -229, 278}
, {-80, -53, -134}
, {-91, -183, -196}
, {-415, -98, -163}
, {239, 95, -147}
, {-339, -194, -367}
, {22, -51, 67}
, {-104, -189, -348}
}
, {{-52, -71, -37}
, {-105, -108, -157}
, {30, 3, -23}
, {-85, 149, -240}
, {-164, 22, -227}
, {-92, -190, -268}
, {-151, -84, -85}
, {-233, -19, -139}
, {-117, -127, -396}
, {3, -84, -33}
, {-93, -240, -52}
, {-14, -119, -195}
, {-152, -44, 5}
, {-339, -23, -113}
, {-118, -40, 23}
, {-242, -162, -110}
}
, {{13, -43, -209}
, {-2, -33, -42}
, {-1, -101, -59}
, {-22, -140, -246}
, {-114, 105, -92}
, {-910, -869, 186}
, {-49, -110, 128}
, {-31, -409, 229}
, {-229, -132, -270}
, {-4, -390, -100}
, {-157, -192, -476}
, {-56, -173, -532}
, {-263, -87, -39}
, {-215, -427, -770}
, {-58, -58, -16}
, {-319, -127, -134}
}
, {{-95, -92, 70}
, {64, -51, -102}
, {-316, 272, -81}
, {-31, 221, 225}
, {39, -102, -57}
, {-567, -90, -22}
, {-35, -51, 121}
, {92, -167, -156}
, {-91, -57, -121}
, {-194, -108, -81}
, {-238, -129, -342}
, {-40, -75, -109}
, {-37, -172, -19}
, {-380, -238, -230}
, {-51, 19, 42}
, {-338, -82, -184}
}
, {{-146, 9, -29}
, {-55, 26, -72}
, {-132, -3, -64}
, {-103, -103, 15}
, {-41, 24, 36}
, {-592, -38, -322}
, {-148, -114, -53}
, {-8, -108, 33}
, {-218, -130, -44}
, {-77, 46, 0}
, {-290, -103, -102}
, {-32, 8, -44}
, {-92, 61, 54}
, {-334, -442, -128}
, {-3, 47, -101}
, {-123, -7, 0}
}
, {{71, 27, -1}
, {-44, -109, 38}
, {-358, -555, -198}
, {-82, 76, -136}
, {-58, 53, -31}
, {-214, 146, -1023}
, {61, 129, -37}
, {279, -129, -162}
, {136, -248, -506}
, {-142, -197, -249}
, {-53, -288, -158}
, {-139, -313, -296}
, {66, 34, -53}
, {-649, -390, -198}
, {-40, -178, 146}
, {-39, -250, -512}
}
, {{113, 51, -23}
, {-71, 84, 68}
, {45, -6, 72}
, {-49, 35, -81}
, {-62, 49, -46}
, {-412, 50, -418}
, {-184, -35, -7}
, {109, -121, 60}
, {-299, -189, 89}
, {-128, -104, 14}
, {-394, -416, -582}
, {-286, -15, -126}
, {23, -94, 23}
, {-144, -486, -131}
, {17, 34, 65}
, {-37, -240, -467}
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

typedef number_t max_pooling1d_3_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_3(
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

typedef number_t average_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

void average_pooling1d(
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

//typedef number_t *flatten_output_type;
typedef number_t flatten_output_type[OUTPUT_DIM];

#define flatten //noop (IN, OUT)  OUT = (number_t*)IN

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

typedef number_t dense_output_type[FC_UNITS];

static inline void dense(
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


const int16_t dense_bias[FC_UNITS] = {-278, -402, 656}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-100, -88, 296, -194, 86, 196, -324, -95, 486, -80, -111, 104, 51, -215, 115, 2, -93, 62, -41, 138, 63, 68, 21, 0, -14, 94, -51, -77, 59, 142, -50, -98}
, {162, -108, 142, 129, -251, -224, -29, -71, -588, 340, 64, -135, 32, -127, 157, 40, -179, 206, 212, -296, 18, -233, 104, -270, -7, 95, -5, -393, -163, 28, 176, -56}
, {7, -102, -187, 0, -188, 43, 219, 19, -84, -118, -173, -78, -39, -140, 212, 59, -5, 279, -213, -25, -13, 74, -219, -157, -35, 32, 48, 90, 7, 40, 7, -21}
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
  //dense_output_type dense_output);
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
