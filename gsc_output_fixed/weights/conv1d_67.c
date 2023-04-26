/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    5
#define CONV_FILTERS      3
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_67_bias[CONV_FILTERS] = {-39, -11, -33}
;

const int16_t conv1d_67_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-139, -211, 173}
, {-215, 267, 176}
, {132, -37, 132}
, {172, -56, 234}
, {244, -50, 252}
}
, {{-173, -6, -200}
, {120, 59, 125}
, {120, -202, -177}
, {162, 127, 73}
, {18, -178, -145}
}
, {{-243, -25, 85}
, {2, -110, -12}
, {-66, 97, 61}
, {231, 279, 130}
, {-189, -92, 84}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE