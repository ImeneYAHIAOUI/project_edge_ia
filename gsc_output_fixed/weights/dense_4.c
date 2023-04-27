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


const int16_t dense_4_bias[FC_UNITS] = {-39, -14, 71}
;

const int16_t dense_4_kernel[FC_UNITS][INPUT_SAMPLES] = {{-560, 65, 173, -222, -187, 70, 165, 10, 90, -108, 169, 136, -79, -58, 135, -141, -60, 84, 96, -212, -12, 106, 2, -11, -105, 51, 111, -110, -109, -157, -211, 290}
, {264, -147, -177, 17, 172, -3, 194, -194, -352, 116, 30, -180, 28, 97, -315, -96, 69, -40, 136, -275, 23, 130, 108, 59, 225, 75, -33, -252, -39, -251, -75, -92}
, {-142, -146, 151, 170, -125, 93, -76, 72, 216, -34, -116, 166, -112, 81, 70, -204, -34, -335, -177, 32, -5, 23, -29, -124, 59, 34, -79, 65, -106, 208, 26, -499}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS