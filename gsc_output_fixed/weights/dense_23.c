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


const int16_t dense_23_bias[FC_UNITS] = {131, -171, 62}
;

const int16_t dense_23_kernel[FC_UNITS][INPUT_SAMPLES] = {{105, 26, 123, -202, 239, 212, -262, -135, -23, 42, 10, -112, 103, -98, -160, -223, 2, -327, 324, -217, 85, 135, -159, -2, 151, -40, 160, -184, -67, -41, 41, -7}
, {-170, -19, 79, 154, -68, -16, -252, 156, -161, 49, -99, 21, 24, 85, 304, 219, 341, 189, -171, 61, 123, -166, 106, 20, 175, -32, 136, -250, 30, 28, -20, 224}
, {4, 13, -117, -184, 222, 125, 344, 108, -176, 55, -123, -219, -105, 354, -148, 1, -73, -313, 226, -38, 2, 38, -105, -55, 116, -84, -13, 170, 226, -80, -128, 111}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS