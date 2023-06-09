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


const int16_t conv1d_25_bias[CONV_FILTERS] = {-89, -198, 42, -121, 208, 158, 92, -121}
;

const int16_t conv1d_25_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-62, -316, -232}
, {-211, 19, -232}
, {-338, -221, -173}
, {164, 66, 1}
, {74, -56, -224}
, {-89, 105, 155}
, {69, -100, -181}
, {4, -32, -185}
}
, {{-145, -151, -141}
, {-130, 10, -81}
, {-120, -159, -188}
, {-98, -6, 37}
, {88, 55, -50}
, {16, -25, 17}
, {120, 35, 239}
, {82, 121, 127}
}
, {{-158, 106, 201}
, {57, -112, 134}
, {-24, 90, 243}
, {-60, -60, 31}
, {-257, -153, 2}
, {197, -209, -315}
, {125, -124, -157}
, {-242, 102, -59}
}
, {{5, 11, 127}
, {251, -17, 240}
, {-37, -362, -91}
, {-64, -144, 117}
, {137, 37, -132}
, {-220, -127, 139}
, {-50, -286, 209}
, {108, -44, 72}
}
, {{-58, -228, -135}
, {-73, -269, -263}
, {-248, -273, -87}
, {-117, 119, 162}
, {-209, -50, 57}
, {44, -60, 296}
, {-248, -63, -470}
, {125, 260, 150}
}
, {{-55, -137, -9}
, {-129, -46, -92}
, {-26, 54, 103}
, {-365, -252, -297}
, {-46, 47, 141}
, {-144, 257, 102}
, {-41, -250, -249}
, {-51, -9, -131}
}
, {{119, -228, 27}
, {-73, -136, 78}
, {64, -61, 55}
, {29, -276, -139}
, {74, -195, -231}
, {-141, 117, 112}
, {-327, -213, -349}
, {27, -2, -191}
}
, {{30, -12, -7}
, {-136, -31, -150}
, {32, 72, 151}
, {135, 157, -58}
, {-104, -166, -110}
, {134, 175, 182}
, {-73, -28, 46}
, {138, -5, 106}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE