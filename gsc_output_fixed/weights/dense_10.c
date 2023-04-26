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


const int16_t dense_10_bias[FC_UNITS] = {-35, 25, 5}
;

const int16_t dense_10_kernel[FC_UNITS][INPUT_SAMPLES] = {{-25, 50, -80, 229, 164, -68, -146, 61, -49, -100, 72, 115, 58, -145, -280, 123, -240, 10, 72, -17, -262, -23, -78, 103, 110, -46, 8, 63, 45, 179, -264, 89}
, {-325, 215, 350, 124, 186, -194, 136, 37, 173, 6, -233, -188, 7, 142, -36, -225, 52, -167, -231, -17, 85, 235, -210, 335, 48, -274, 107, -160, -190, -188, -157, 0}
, {72, -106, -140, -96, -139, 151, 200, -22, -152, -67, 159, 17, 20, 278, 177, 264, 39, -107, 188, -269, 86, 187, 8, -492, 123, 239, 16, -425, -63, -263, 190, -48}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS