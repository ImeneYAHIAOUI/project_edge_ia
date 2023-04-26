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


const int16_t dense_1_bias[FC_UNITS] = {4, -7, 13}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{185, 4, 135, 82, 41, -89, 300, -239, -302, 30, 354, -61, 295, 279, -33, -173, -42, 65, 181, -82, -158, -21, -172, -234, -20, 319, -23, 484, -89, -75, -540, 88}
, {-323, 411, 179, -268, 125, -398, -343, 277, 31, 45, -406, -346, -467, -83, -165, -45, 117, -81, -145, -131, 152, 155, 96, -332, 149, -208, 122, -193, -2, 172, 259, 67}
, {118, -376, 97, 64, 137, 269, -55, -208, 65, -28, 216, 161, -90, -94, 162, -134, -227, -134, -121, 124, -168, -64, -98, 297, -109, -20, -26, -293, 393, 212, 32, -88}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS