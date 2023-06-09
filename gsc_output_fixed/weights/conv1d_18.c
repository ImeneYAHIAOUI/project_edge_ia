/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_18_bias[CONV_FILTERS] = {31, -144, 180, -5, -102, 56, -22, 161, 155, -170, -88, 77, -336, -99, 266, 165}
;

const int16_t conv1d_18_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{122, -55, 26}
, {-79, -172, -20}
, {-120, -94, -131}
, {-159, 102, -81}
, {204, 223, 289}
, {98, 114, 140}
, {-256, -181, -231}
, {147, -105, -106}
}
, {{55, -222, -24}
, {-28, -36, 6}
, {-154, -186, -186}
, {147, 282, 128}
, {-97, -17, -6}
, {215, 106, -110}
, {-3, -225, 55}
, {-44, -75, -122}
}
, {{154, 102, 10}
, {-320, 200, -85}
, {85, 48, 25}
, {-165, -113, 143}
, {-97, -12, 118}
, {31, -184, -34}
, {-232, -64, -234}
, {53, -17, 49}
}
, {{95, 38, 44}
, {-31, -89, -206}
, {264, -24, 61}
, {-234, -38, -161}
, {80, 96, -178}
, {-20, -139, -72}
, {78, -119, 73}
, {-56, 225, -125}
}
, {{-4, -217, -148}
, {-54, -168, 32}
, {-79, 9, -11}
, {-183, -154, -168}
, {-9, -128, -72}
, {-255, -132, -236}
, {-3, 43, -185}
, {-218, -67, 19}
}
, {{80, -113, -8}
, {-26, -50, 167}
, {-177, 102, -80}
, {-56, -152, -137}
, {-212, -98, -11}
, {71, -151, -276}
, {205, -114, -165}
, {113, 17, -75}
}
, {{-185, 84, -18}
, {-5, -135, -94}
, {-106, -67, 19}
, {-148, -133, -48}
, {-78, -118, -67}
, {-23, 35, -26}
, {-6, -33, -88}
, {-33, -41, -90}
}
, {{11, -23, 182}
, {-211, -268, -189}
, {-255, -281, -323}
, {-126, -150, -22}
, {-47, 158, 287}
, {34, -1, 129}
, {158, -45, -82}
, {-27, 87, 167}
}
, {{65, -64, 304}
, {170, -112, -194}
, {31, -113, 46}
, {76, -152, -321}
, {5, 131, -45}
, {155, -45, -212}
, {-224, 36, -163}
, {-7, 112, 89}
}
, {{-363, -285, -289}
, {-298, -298, -87}
, {-72, -21, 145}
, {67, 69, -54}
, {71, 32, -126}
, {106, 232, 42}
, {67, 3, 19}
, {-296, -94, 45}
}
, {{165, -138, -160}
, {-281, 34, -212}
, {80, -185, 19}
, {196, -231, 79}
, {-204, 135, -37}
, {-192, -113, -60}
, {-236, 101, 160}
, {-19, -214, 8}
}
, {{92, 39, -213}
, {-69, -284, -116}
, {-105, -226, -185}
, {-340, -129, -192}
, {-13, 268, 179}
, {86, 126, -45}
, {-38, 68, 26}
, {140, -13, -196}
}
, {{-2, 212, -31}
, {96, 35, -17}
, {-99, 7, 158}
, {100, 12, 174}
, {-86, 66, -186}
, {161, -74, -6}
, {174, -61, -138}
, {201, 81, -6}
}
, {{-320, -232, -145}
, {46, -19, 15}
, {163, 41, 66}
, {-297, -181, -196}
, {-128, -162, 9}
, {-279, -314, -173}
, {94, 93, -85}
, {-207, -251, -269}
}
, {{12, -133, 68}
, {157, 165, -55}
, {110, 141, 174}
, {-160, -32, -27}
, {-12, 164, -153}
, {19, 74, 19}
, {-423, -322, -302}
, {-228, -131, -96}
}
, {{95, -206, -21}
, {-332, -213, -114}
, {190, -56, -145}
, {-177, -266, -336}
, {-17, -7, -218}
, {-42, 13, -32}
, {81, 22, 95}
, {-254, -251, 68}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE