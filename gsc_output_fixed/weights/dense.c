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


const int16_t dense_bias[FC_UNITS] = {-278, -402, 656}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-100, -88, 296, -194, 86, 196, -324, -95, 486, -80, -111, 104, 51, -215, 115, 2, -93, 62, -41, 138, 63, 68, 21, 0, -14, 94, -51, -77, 59, 142, -50, -98}
, {162, -108, 142, 129, -251, -224, -29, -71, -588, 340, 64, -135, 32, -127, 157, 40, -179, 206, 212, -296, 18, -233, 104, -270, -7, 95, -5, -393, -163, 28, 176, -56}
, {7, -102, -187, 0, -188, 43, 219, 19, -84, -118, -173, -78, -39, -140, 212, 59, -5, 279, -213, -25, -13, 74, -219, -157, -35, 32, 48, 90, 7, 40, 7, -21}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS