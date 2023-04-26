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


const int16_t dense_1_bias[FC_UNITS] = {-11, 86, -67}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{59, 145, -137, -148, -53, 72, -30, 15, 173, -140, 73, -1, 2, -65, -175, -30, 80, -196, 43, 2, 47, -63, -133, 240, -129, -89, -231, 35, -106, 79, -112, -74}
, {184, 108, -54, -219, -81, 46, -76, -151, -4, -74, -111, 70, -95, 176, -104, 108, -305, 251, -96, -51, 196, 87, -204, -345, 2, 91, 167, 53, 0, 160, -5, -115}
, {94, -45, 13, 28, -6, 62, 82, -131, -95, -150, 43, 53, -70, -4, 163, 64, -93, 51, -85, 57, -191, -135, -67, 251, -43, -126, -261, -135, -128, -250, -60, -99}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS