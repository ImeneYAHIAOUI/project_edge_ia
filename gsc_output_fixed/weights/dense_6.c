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


const int16_t dense_6_bias[FC_UNITS] = {7, -53, 40}
;

const int16_t dense_6_kernel[FC_UNITS][INPUT_SAMPLES] = {{-81, 95, -127, -79, -42, 44, 74, 63, -21, 117, 20, -347, 236, -88, -80, 20, -39, -71, 45, 29, -142, 26, -166, -3, 12, -8, -81, -142, 9, 5, 118, -154}
, {96, -33, 99, -40, -269, 120, -237, -29, 89, 114, 51, 319, -371, -157, 25, 9, -152, 150, -84, 137, -77, 330, -48, 65, 168, -186, 87, -143, 89, -85, -51, 4}
, {-28, -21, -40, -70, 60, -215, 84, 50, -45, -151, -124, -321, -33, 71, -53, 227, -118, 57, 17, 56, -51, 130, -73, 18, -158, 204, 102, 101, 32, -24, 88, 48}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS