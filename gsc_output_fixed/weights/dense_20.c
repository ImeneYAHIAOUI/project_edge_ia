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


const int16_t dense_20_bias[FC_UNITS] = {7, -67, 34}
;

const int16_t dense_20_kernel[FC_UNITS][INPUT_SAMPLES] = {{-80, -124, -70, -149, -136, -95, 74, 134, -132, 4, -176, 2, 86, -82, -102, 136, 209, -89, 163, -56, -7, 12, 215, 98, 68, -265, -102, 117, 292, 142, 252, -47}
, {-31, -180, 39, 137, 166, -62, 132, -112, -61, -167, 39, -6, -39, -32, -204, 53, 163, 28, -180, 28, -28, 172, 148, 26, 22, 78, -211, -233, -122, -133, -86, 3}
, {47, -33, 161, 82, -37, 56, -121, -104, -112, 234, -5, 146, 100, 56, 31, 15, -67, -347, 186, -181, -157, 35, -131, -191, -271, 299, 270, 83, -149, -88, -49, 54}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS