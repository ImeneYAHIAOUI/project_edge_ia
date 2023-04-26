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


const int16_t dense_24_bias[FC_UNITS] = {138, -72, -63}
;

const int16_t dense_24_kernel[FC_UNITS][INPUT_SAMPLES] = {{-4, -60, 189, 61, -150, 93, 138, 172, -323, -225, 10, -130, -153, 169, 77, -257, 38, 110, 145, -199, -194, -216, -3, 63, -121, 56, -302, -88, -341, 80, -7, -218}
, {-160, -22, -173, 73, -169, -18, 126, -132, 107, -26, -122, -57, 109, -127, 69, 66, 26, 181, 125, -63, -5, -12, 82, 143, 152, 94, -244, 12, 209, -343, -284, -2}
, {-35, 119, -336, 23, 151, -49, -48, 84, 21, 28, 160, 23, 73, 45, -17, 42, -227, -132, -123, -13, -84, 160, 30, 71, 136, -79, 208, -152, 238, 283, 126, 153}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS