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


const int16_t dense_bias[FC_UNITS] = {2, -44, 38}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-51, -90, 165, -74, -24, 82, -2, 134, 125, -18, 36, -14, -225, -103, 37, 159, -191, -204, -113, -86, 115, 31, 126, 79, 49, -115, -96, -43, -28, 258, -56, 50}
, {186, -33, -22, 89, -19, -286, -79, -296, 100, -37, 187, 11, 22, -131, 61, 140, 12, 262, 220, -16, 48, -209, 96, 120, -49, 133, -159, 125, -106, -407, -174, -26}
, {35, -56, -7, 106, -84, 201, 61, 51, 66, -131, 33, -80, -61, -53, 105, -5, 104, -58, -260, 77, 158, 147, 114, 76, 111, -174, -116, 56, 20, 48, 18, 190}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS