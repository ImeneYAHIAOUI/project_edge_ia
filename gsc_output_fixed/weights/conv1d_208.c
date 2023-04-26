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


const int16_t conv1d_208_bias[CONV_FILTERS] = {38, -269, 33, 62, -114, -228, -6, -31}
;

const int16_t conv1d_208_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-248, 145, -32}
, {28, 39, 144}
, {69, -22, 24}
, {-228, -103, -43}
, {207, 6, 214}
, {-3, -100, -114}
, {87, 95, -119}
, {196, -2, -15}
}
, {{65, 105, 120}
, {58, 46, 54}
, {107, 143, 29}
, {-46, 95, 87}
, {132, 180, -4}
, {-108, -285, -265}
, {119, 89, -174}
, {66, 333, -5}
}
, {{-2, -16, 215}
, {-107, 99, 180}
, {-147, -2, -147}
, {-72, 112, 382}
, {-269, 15, 79}
, {-79, -5, 136}
, {-141, -160, 198}
, {-229, 41, -174}
}
, {{158, 112, -111}
, {1, 165, -220}
, {176, 193, 209}
, {144, -59, -107}
, {-85, 211, -103}
, {-284, -365, 36}
, {-87, 61, -211}
, {-11, 109, 2}
}
, {{88, -52, -264}
, {-98, -88, -156}
, {116, 73, 98}
, {95, -201, -61}
, {-67, 255, -208}
, {-48, -3, 98}
, {71, -56, -5}
, {234, 120, -109}
}
, {{40, 131, 108}
, {-103, -22, 61}
, {-145, 9, -296}
, {-35, 163, 118}
, {6, -105, -185}
, {-125, 124, 146}
, {-205, -167, 96}
, {246, 124, -178}
}
, {{64, 99, -88}
, {-188, -82, 155}
, {-5, -36, -110}
, {-30, 6, -130}
, {51, -28, 139}
, {-199, -259, -271}
, {-2, -56, -160}
, {-171, -38, 225}
}
, {{-163, -98, -120}
, {-56, -123, -23}
, {-187, -227, -250}
, {94, -4, -104}
, {-34, 31, -74}
, {-265, -256, -163}
, {22, -38, 83}
, {97, 93, 16}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE