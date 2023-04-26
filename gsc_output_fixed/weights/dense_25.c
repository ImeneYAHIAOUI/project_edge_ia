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


const int16_t dense_25_bias[FC_UNITS] = {56, -18, -16}
;

const int16_t dense_25_kernel[FC_UNITS][INPUT_SAMPLES] = {{-118, 129, -58, -40, 34, -120, -147, -75, 72, -74, -153, -67, -217, 10, -207, -169, 71, 71, 40, 70, 138, -7, -56, -52, 170, 210, 66, -85, -204, 144, -244, 113}
, {-67, 197, -44, 113, 11, -183, 177, 134, -88, -233, -103, -64, 24, -83, -129, 265, -172, 131, -8, -162, -61, 111, 48, 95, 243, -88, -226, -45, 255, 214, -56, -63}
, {245, -328, 180, -60, -163, 234, 206, -187, 140, 291, 257, -262, -219, -42, -143, 29, -123, -134, 259, -97, -75, 105, 46, 117, -167, -260, 85, -35, -64, -43, -244, -416}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS