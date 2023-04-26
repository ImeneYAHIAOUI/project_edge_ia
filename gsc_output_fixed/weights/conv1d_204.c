/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_204_bias[CONV_FILTERS] = {-118, 21, -47, -45, 65, -70, 3, 188}
;

const int16_t conv1d_204_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{110, 171, 18}
, {-88, -20, -83}
, {71, -155, -101}
, {27, 80, 110}
, {62, 83, 132}
, {-250, -20, -13}
, {144, 87, 195}
, {-181, 34, -38}
}
, {{-368, -98, -132}
, {-7, 27, 148}
, {4, -49, -304}
, {35, 199, 192}
, {0, 197, 25}
, {36, -204, -44}
, {93, 236, -56}
, {-112, 179, 119}
}
, {{114, -246, -237}
, {-65, 62, -1}
, {-126, 81, -191}
, {-12, -89, -56}
, {-60, -173, 61}
, {-48, 99, -140}
, {-277, -122, 169}
, {100, -127, -215}
}
, {{-117, -188, 141}
, {-213, 56, -118}
, {-43, -84, -206}
, {-37, 35, -61}
, {-18, -80, -189}
, {16, -217, -126}
, {-53, -239, -6}
, {-178, -7, -260}
}
, {{-24, 23, 211}
, {52, 41, 13}
, {-124, 74, 81}
, {-354, -182, -206}
, {-350, -84, -11}
, {-267, -143, 200}
, {184, 11, 110}
, {-232, -303, -496}
}
, {{170, -119, 157}
, {-68, -164, -141}
, {-4, 213, 199}
, {180, 27, -127}
, {-2, -283, -200}
, {60, 215, -60}
, {-105, -105, -9}
, {-14, 77, 32}
}
, {{142, -46, -266}
, {-193, -148, -138}
, {-60, -284, -104}
, {96, -147, 8}
, {-30, 13, -265}
, {-9, 15, 36}
, {98, -154, 65}
, {-63, 101, -146}
}
, {{-158, -117, -22}
, {183, 95, -31}
, {114, -184, -115}
, {-72, 12, 186}
, {169, 172, 98}
, {133, -90, 32}
, {-241, -17, -115}
, {26, 0, 129}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE