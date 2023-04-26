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


const int16_t dense_17_bias[FC_UNITS] = {-80, -47, 128}
;

const int16_t dense_17_kernel[FC_UNITS][INPUT_SAMPLES] = {{34, -29, 179, -131, 125, 123, 44, -77, 111, -174, -101, -36, 38, -178, 71, 106, -250, -218, -146, 35, -85, 177, 0, 37, 180, -273, -96, 2, 86, -27, 55, 28}
, {-143, -144, 150, -84, 208, 127, -167, 190, -33, -74, 103, -176, -317, -80, 85, -150, -17, 183, 171, -183, 95, -68, -15, 181, -65, -264, 103, -399, 154, -30, 114, 19}
, {79, -28, -307, 75, -155, -121, -92, 134, -7, 164, -28, 115, 77, 326, 48, -64, 200, -110, -37, 316, 5, 120, -56, 65, -128, 417, 155, 174, 132, -164, -63, 93}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS