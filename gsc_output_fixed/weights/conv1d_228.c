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


const int16_t conv1d_228_bias[CONV_FILTERS] = {34, -59, -52, -65, -114, 14, -68, -70}
;

const int16_t conv1d_228_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-202, -100, 99}
, {-69, -288, 201}
, {-144, -1, -180}
, {52, -178, 185}
, {19, -5, 55}
, {-168, -235, -57}
, {-230, -166, -125}
, {-164, -39, -73}
}
, {{169, -39, -118}
, {-124, -17, 60}
, {-54, -203, -301}
, {94, 149, 42}
, {-118, 43, -214}
, {85, 166, -141}
, {161, 14, -115}
, {174, 45, -30}
}
, {{-1, -32, -274}
, {120, -176, 85}
, {5, -177, -162}
, {-1, -238, 53}
, {-97, 50, -167}
, {-279, 78, 115}
, {169, -138, -247}
, {75, -201, -263}
}
, {{-12, -112, -85}
, {-29, 103, 141}
, {-185, -84, -96}
, {-221, -437, -335}
, {191, 154, 52}
, {-224, -66, -257}
, {73, 65, 33}
, {-50, 87, 5}
}
, {{-241, -157, -96}
, {44, 198, 236}
, {27, -208, -287}
, {-94, -47, -125}
, {55, 114, 17}
, {6, 30, 118}
, {-146, -150, -298}
, {-164, -43, -135}
}
, {{75, -127, -14}
, {-166, 131, -42}
, {266, 71, 94}
, {-31, -101, -98}
, {134, 67, 195}
, {209, -76, 203}
, {-231, -89, -337}
, {-56, -131, -289}
}
, {{158, 61, -113}
, {-226, 156, -148}
, {-105, 71, 150}
, {97, 155, 166}
, {-227, -100, -196}
, {62, 71, 75}
, {-140, -243, 45}
, {-135, -75, 86}
}
, {{-145, -76, -204}
, {-279, 140, 207}
, {105, -109, -169}
, {42, 29, 182}
, {-201, 99, -162}
, {-28, -157, -113}
, {-235, -132, 71}
, {-436, -110, 107}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE