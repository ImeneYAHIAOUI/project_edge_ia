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


const int16_t conv1d_245_bias[CONV_FILTERS] = {-100, -95, -63, -45, -13, -180, -161, -16, -13, -170, 90, 87, -9, -202, -235, -190}
;

const int16_t conv1d_245_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{201, 294, 143}
, {66, 162, 109}
, {175, 147, -117}
, {-40, -123, 27}
, {211, 147, 28}
, {-186, -42, 7}
, {81, -208, 3}
, {-135, 115, -82}
}
, {{29, -167, -217}
, {48, -173, 48}
, {-1, 78, -157}
, {50, -14, 185}
, {-257, 9, -37}
, {-84, 37, 292}
, {78, -198, 126}
, {-183, -22, -312}
}
, {{48, -13, -94}
, {-124, -130, -114}
, {14, 0, -240}
, {-51, -165, 14}
, {-145, -154, -3}
, {-162, -9, 30}
, {-6, 41, -164}
, {-48, -144, 71}
}
, {{-1, 27, -65}
, {56, -76, -135}
, {-183, -82, 44}
, {-152, -113, 18}
, {-140, 49, -38}
, {-72, -192, -10}
, {-128, -131, -129}
, {-139, -68, -115}
}
, {{90, -18, -41}
, {-99, 55, 10}
, {-67, 33, 86}
, {-232, -35, -64}
, {0, 87, 145}
, {-130, 14, -122}
, {-142, -183, 34}
, {26, 136, 78}
}
, {{276, 3, -106}
, {-61, -13, 60}
, {-9, -68, 79}
, {-178, 145, 35}
, {27, -63, -37}
, {164, 143, 75}
, {23, 54, -76}
, {168, -18, -173}
}
, {{128, 175, -274}
, {37, -64, -193}
, {59, 92, 1}
, {-75, 53, 43}
, {-132, -117, 2}
, {-14, 364, -28}
, {-54, -4, 6}
, {241, -11, -31}
}
, {{-269, -184, -323}
, {-157, -130, -194}
, {27, -134, -85}
, {78, 111, 1}
, {-24, 20, -39}
, {-77, 216, -86}
, {196, 99, 399}
, {-205, -149, 101}
}
, {{-51, 154, -46}
, {118, 119, 76}
, {-61, -64, -79}
, {45, 57, 171}
, {92, 87, 191}
, {-239, 265, -41}
, {-197, -192, -192}
, {45, -182, 79}
}
, {{-203, -100, 179}
, {40, -134, -265}
, {133, -13, -225}
, {-171, 120, 136}
, {37, 136, -47}
, {240, 167, -23}
, {331, -4, 81}
, {-155, 72, -11}
}
, {{-26, -144, 50}
, {42, -8, 25}
, {99, -17, 97}
, {-191, -26, 26}
, {-175, -149, -201}
, {-53, -185, 75}
, {-57, 38, 135}
, {-41, -191, -202}
}
, {{63, -110, -2}
, {-30, 41, -355}
, {-75, -420, -202}
, {85, 56, -12}
, {175, -13, -324}
, {59, 79, -115}
, {48, -244, -79}
, {164, 158, 17}
}
, {{-174, -188, -25}
, {63, -231, 45}
, {-25, -130, -179}
, {-150, 61, -87}
, {36, -205, -98}
, {413, -117, 249}
, {273, -270, 33}
, {24, 86, 19}
}
, {{-59, 164, -130}
, {168, 194, -223}
, {144, -103, -162}
, {-67, 35, -105}
, {-197, -103, -13}
, {254, 2, 84}
, {87, -143, -46}
, {26, 182, 52}
}
, {{-196, -151, 0}
, {-21, -199, 5}
, {-201, 7, -262}
, {82, 86, -114}
, {9, -166, -297}
, {136, 14, -131}
, {-53, 26, -41}
, {39, -220, -184}
}
, {{-19, -147, -91}
, {-139, 137, 113}
, {-192, -137, 79}
, {-185, 43, -64}
, {-77, 32, 57}
, {243, -49, 78}
, {165, -36, -241}
, {-173, 24, 93}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE