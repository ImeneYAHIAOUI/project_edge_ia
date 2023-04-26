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


const int16_t conv1d_17_bias[CONV_FILTERS] = {5, -38, -38, 68, -28, -17, 96, -59}
;

const int16_t conv1d_17_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-100, -149, -113}
, {-23, -9, 34}
, {-104, -175, 23}
, {50, -52, 13}
, {-180, 96, 55}
, {76, -42, -108}
, {161, 62, 101}
, {-151, -46, -124}
}
, {{-134, -109, -79}
, {45, 103, 81}
, {-18, -61, -217}
, {114, 51, -21}
, {-14, 37, -113}
, {-227, 80, 9}
, {-26, 74, 79}
, {35, -143, -76}
}
, {{-161, -5, 32}
, {100, -157, -51}
, {-8, -75, 3}
, {2, -200, 92}
, {-100, -26, -30}
, {25, -118, -23}
, {-228, 51, -47}
, {4, 68, 92}
}
, {{92, 90, 91}
, {-62, 81, 2}
, {-147, -96, -81}
, {131, -159, -34}
, {62, -7, -25}
, {122, -22, 38}
, {91, -22, 153}
, {-4, 19, -37}
}
, {{26, -64, -3}
, {-19, -3, 165}
, {99, -167, -110}
, {-150, -73, -72}
, {-165, -16, -170}
, {-156, -152, -227}
, {27, 124, -132}
, {152, 39, -66}
}
, {{-22, -78, -80}
, {-74, 58, -90}
, {185, -136, 1}
, {63, 110, 130}
, {173, -135, -22}
, {54, 33, -112}
, {-96, -134, -141}
, {-191, -155, -164}
}
, {{-111, -72, 58}
, {-206, -83, -3}
, {-10, 147, -23}
, {-153, -176, -91}
, {-47, 134, -23}
, {126, -159, 154}
, {-22, 135, 141}
, {-28, 46, -36}
}
, {{-107, 18, 91}
, {-38, -134, -95}
, {51, 91, -139}
, {-107, 25, -165}
, {-2, -56, -17}
, {-137, 50, -163}
, {-17, 5, 143}
, {-91, -17, 104}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE