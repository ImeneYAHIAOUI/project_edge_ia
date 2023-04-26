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


const int16_t dense_14_bias[FC_UNITS] = {35, -109, 120}
;

const int16_t dense_14_kernel[FC_UNITS][INPUT_SAMPLES] = {{-102, -146, 89, 176, -136, 64, 114, -140, 48, 74, -165, 87, 74, 115, 131, -142, -29, 26, 56, -50, -100, -144, -98, -136, -11, 21, 184, 128, 21, -139, 149, 76}
, {-170, -162, 29, 87, -33, 78, 161, 13, 101, -151, -144, -44, 164, -52, 147, -147, -194, 61, -96, -6, -79, 36, 35, -66, 182, 50, 35, 38, 143, 94, -49, -73}
, {78, -164, -95, -170, -110, 148, 44, 68, 110, 64, 93, -157, -34, 57, 129, 185, -56, -55, -7, 149, -111, -66, 159, 174, -125, -155, -206, 92, 88, -6, -169, 41}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS