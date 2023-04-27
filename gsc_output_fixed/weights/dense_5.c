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