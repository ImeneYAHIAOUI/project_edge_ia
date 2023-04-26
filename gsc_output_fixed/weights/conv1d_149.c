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


const int16_t conv1d_149_bias[CONV_FILTERS] = {-75, -17, 87, -76, -84, -72, 19, 61}
;

const int16_t conv1d_149_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-74, -185, 82}
, {-31, -136, -206}
, {-36, -13, 68}
, {-72, -76, -143}
, {-47, 38, 61}
, {71, 15, -196}
, {-186, 43, -223}
, {96, -145, 71}
}
, {{-63, -10, -53}
, {-63, 54, 82}
, {61, -30, -42}
, {22, -86, -145}
, {155, 0, 12}
, {-45, -194, -152}
, {84, -67, -195}
, {26, -156, -80}
}
, {{-44, 97, 125}
, {-27, 106, 6}
, {-40, 209, 78}
, {-14, -4, 107}
, {149, -30, 60}
, {-140, 6, -100}
, {-30, 105, 16}
, {145, -59, 63}
}
, {{26, 0, 0}
, {63, -143, 29}
, {62, 31, -181}
, {1, -1, 154}
, {55, 102, -147}
, {94, -143, -156}
, {95, 102, -171}
, {117, -127, 77}
}
, {{129, -162, -42}
, {49, 2, 12}
, {-143, 78, -169}
, {-119, 6, -110}
, {-224, -104, -65}
, {-195, 115, -216}
, {-98, 58, -39}
, {-125, -31, -47}
}
, {{61, -220, -83}
, {-125, 71, -215}
, {108, 114, -172}
, {81, 96, -15}
, {51, 84, -50}
, {-125, -103, 102}
, {93, -223, -177}
, {-225, -17, -159}
}
, {{-129, -67, -187}
, {39, 17, 55}
, {-38, 35, -154}
, {33, 135, 129}
, {-63, -68, -12}
, {-51, -134, -89}
, {-84, 100, 98}
, {-190, -184, -201}
}
, {{129, 40, -141}
, {-43, -34, 63}
, {121, 57, -93}
, {160, 114, -152}
, {-147, 120, 132}
, {40, 66, -6}
, {-173, 0, 83}
, {-152, -163, -136}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE