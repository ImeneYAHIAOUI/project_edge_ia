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


const int16_t conv1d_130_bias[CONV_FILTERS] = {40, -39, -44, 30, -9, -71, -47, -59, -36, -30, -6, -50, -10, -59, -25, 30}
;

const int16_t conv1d_130_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{19, -125, -190}
, {-182, -47, 23}
, {-3, 93, 44}
, {-87, -88, 36}
, {-14, -103, -193}
, {-60, 81, 63}
, {64, -10, 44}
, {-45, 143, 21}
}
, {{-30, -156, 83}
, {-211, 46, -5}
, {-126, -202, -96}
, {-214, 38, 55}
, {70, -141, 24}
, {57, 14, -62}
, {-101, -94, -72}
, {-188, -178, 55}
}
, {{29, 65, -162}
, {-39, 59, -95}
, {5, -102, -68}
, {-112, -189, -164}
, {33, -42, 46}
, {-78, 61, 68}
, {43, -80, 11}
, {-195, -7, -135}
}
, {{21, 77, 88}
, {74, 49, 0}
, {-18, -64, -74}
, {128, -9, -18}
, {131, 23, -96}
, {101, 80, -16}
, {36, -114, -127}
, {-136, -50, -72}
}
, {{-3, -2, 13}
, {-70, -24, -17}
, {75, -8, -40}
, {136, -65, -166}
, {-141, 40, 42}
, {80, -172, 87}
, {-2, -3, 27}
, {-49, 136, 97}
}
, {{-13, -109, -211}
, {44, -45, 118}
, {90, -127, -189}
, {48, -86, -29}
, {150, -4, -112}
, {185, 96, -19}
, {-91, -44, 119}
, {-133, 92, -10}
}
, {{-12, 41, -77}
, {107, -78, -157}
, {-70, 32, 36}
, {-33, -31, 113}
, {-132, -34, 25}
, {-109, -186, -60}
, {39, 10, -91}
, {-170, 112, 39}
}
, {{135, 110, -163}
, {133, -132, -128}
, {33, -140, 93}
, {-31, -5, -20}
, {105, -112, 55}
, {-33, 46, -123}
, {102, 75, 3}
, {-16, 89, -151}
}
, {{-65, 18, 30}
, {-60, -144, 48}
, {34, -42, -87}
, {-80, -44, -45}
, {46, 41, -90}
, {-176, 16, -18}
, {-46, 107, -190}
, {-120, -169, 100}
}
, {{-47, 86, 6}
, {9, -7, 90}
, {51, -110, -161}
, {-114, 12, -184}
, {-33, 1, -44}
, {-113, 51, -177}
, {-43, -106, -123}
, {35, -4, -157}
}
, {{-36, 100, -58}
, {2, -183, 79}
, {-47, -33, 14}
, {-99, -143, -107}
, {84, -30, -65}
, {38, -86, -11}
, {5, 139, 5}
, {-147, -68, -147}
}
, {{-9, -62, -28}
, {-120, -51, -189}
, {88, -150, 29}
, {23, 103, -76}
, {48, -131, -116}
, {50, 69, -88}
, {76, 15, -144}
, {-19, -157, -8}
}
, {{157, -31, -18}
, {89, -122, -87}
, {178, -123, 5}
, {-93, 35, -2}
, {-70, -183, 35}
, {79, -168, 72}
, {-67, 33, 82}
, {-101, 55, -161}
}
, {{23, 100, -112}
, {-39, -142, 34}
, {-169, 2, 55}
, {58, 53, -9}
, {54, -156, 57}
, {-164, -105, -126}
, {52, -23, -79}
, {-117, -100, 0}
}
, {{90, -165, 30}
, {-54, -148, -73}
, {-72, 13, -153}
, {-111, -48, 35}
, {19, 69, 13}
, {-1, -175, 51}
, {-72, 100, -165}
, {-123, -129, -103}
}
, {{-140, 89, -122}
, {-9, -26, 115}
, {68, 81, -2}
, {18, -26, -71}
, {-14, 112, -70}
, {1, 52, 0}
, {-123, -10, -126}
, {117, -35, 81}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE