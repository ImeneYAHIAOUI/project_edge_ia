/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_246_bias[CONV_FILTERS] = {-33, 130, -4, -30, -76, -218, -95, -47, -44, -135, -82, 27, 62, -59, 0, 86}
;

const int16_t conv1d_246_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{165, 96, 33}
, {69, -9, -29}
, {-59, 27, -79}
, {81, -125, -62}
, {-89, -194, 45}
, {-114, 56, -15}
, {121, 195, 44}
, {49, -217, -60}
, {-34, -7, 95}
, {120, -224, 7}
, {-40, 58, -25}
, {-446, -92, 69}
, {12, -15, -39}
, {109, 92, -262}
, {-145, -30, -27}
, {11, -72, -54}
}
, {{-109, -18, -46}
, {-109, 46, 142}
, {-28, -190, -30}
, {14, 17, -35}
, {-1, 78, -69}
, {78, -40, -53}
, {12, -66, 85}
, {-30, -196, 32}
, {-167, -127, -263}
, {-283, -220, -3}
, {86, 10, 108}
, {-7, 202, 101}
, {49, -7, 72}
, {-52, -122, 14}
, {19, 16, 30}
, {1, -77, -97}
}
, {{-4, 175, 110}
, {250, -33, -152}
, {-78, 54, 85}
, {76, 58, -16}
, {15, 157, -84}
, {59, 105, -19}
, {63, -42, 84}
, {109, 39, -64}
, {-21, 81, 103}
, {-12, -94, 125}
, {-21, 37, -60}
, {-226, -428, -198}
, {4, 198, -112}
, {-58, -41, 218}
, {-202, -36, -182}
, {-1, 174, -73}
}
, {{-85, 39, -41}
, {45, -69, 54}
, {63, 21, 121}
, {-57, 20, -102}
, {-6, 53, -139}
, {26, 35, 79}
, {246, 200, -109}
, {-72, -199, 95}
, {-23, -23, -78}
, {-104, -33, 116}
, {-5, 88, 28}
, {110, 135, 69}
, {101, 42, -6}
, {183, -6, -191}
, {-28, -40, 90}
, {93, -68, -166}
}
, {{113, 100, 38}
, {-226, -113, -48}
, {9, -41, -98}
, {108, 30, -72}
, {72, -98, -65}
, {141, 158, -53}
, {123, 189, 118}
, {-137, 86, -9}
, {-84, -131, -67}
, {-204, -55, 18}
, {-144, 49, 127}
, {-48, -38, 133}
, {57, -183, -89}
, {89, 29, 38}
, {92, 46, -92}
, {-189, -69, 108}
}
, {{6, -72, -124}
, {70, -123, 48}
, {43, -30, -6}
, {64, 0, -50}
, {97, -80, 245}
, {38, 185, -80}
, {171, 120, -72}
, {-47, -219, -174}
, {88, -120, 21}
, {149, -31, -47}
, {-219, 52, -61}
, {-65, 40, 26}
, {-190, 179, 18}
, {-135, 123, 24}
, {10, 13, 126}
, {-129, 56, 52}
}
, {{14, -222, -224}
, {-124, 81, -104}
, {-78, 35, 6}
, {-35, 23, -102}
, {9, -134, -42}
, {-95, -77, -89}
, {-125, -59, -135}
, {-50, -34, -64}
, {-157, -10, -35}
, {-113, 3, -23}
, {-58, -88, -54}
, {-39, -2, 60}
, {-32, 4, -76}
, {-51, -193, 7}
, {-52, 11, 174}
, {-141, 14, -112}
}
, {{111, -44, -103}
, {77, -96, 107}
, {63, 59, 9}
, {28, -116, 46}
, {-36, 72, -33}
, {-16, 54, -21}
, {50, 4, -17}
, {-122, -132, -146}
, {-125, -214, -212}
, {94, 47, -19}
, {80, -42, 35}
, {20, 26, 26}
, {70, 93, 16}
, {111, 141, -61}
, {-127, 21, -35}
, {-85, -31, -71}
}
, {{-96, -182, -100}
, {39, 329, 129}
, {-7, 138, -114}
, {-41, 0, -60}
, {-80, 31, -248}
, {48, 345, -168}
, {76, 15, 118}
, {-62, 186, 265}
, {-27, 102, 45}
, {33, 18, 287}
, {-210, -39, -28}
, {124, -206, 7}
, {108, -88, -16}
, {-161, 81, -104}
, {139, 30, -29}
, {57, -251, -55}
}
, {{-94, 10, -129}
, {-124, -40, -136}
, {-76, -99, -162}
, {36, 61, 115}
, {-46, -313, 22}
, {-136, -128, 117}
, {19, -33, 74}
, {-219, -140, -85}
, {15, 33, 3}
, {-33, -79, -189}
, {-227, -236, 185}
, {59, -47, 19}
, {54, -37, -73}
, {304, 16, -130}
, {-145, 123, -105}
, {203, -154, 40}
}
, {{-211, -165, -186}
, {28, -45, 29}
, {-11, 106, -166}
, {-49, 55, -81}
, {-100, -144, 184}
, {213, -43, -176}
, {161, 10, -154}
, {-32, 138, -52}
, {111, 31, -92}
, {-103, -213, -82}
, {-119, 69, -200}
, {30, -104, -46}
, {49, -49, 20}
, {-117, 61, -67}
, {-4, -57, -183}
, {-144, -37, 0}
}
, {{104, -139, 18}
, {-109, -53, 88}
, {36, 143, 56}
, {12, -92, -83}
, {-63, 34, -112}
, {-116, -32, -116}
, {-152, -19, -94}
, {-130, -199, -97}
, {-188, -144, -233}
, {-99, -30, -201}
, {66, 64, -83}
, {-257, -156, -90}
, {-1, -13, 68}
, {23, -68, -66}
, {108, -63, -64}
, {-50, 88, -39}
}
, {{-134, -44, -131}
, {-108, -65, -237}
, {-12, -37, 120}
, {-29, -166, -143}
, {-200, -104, 89}
, {-43, -61, 51}
, {-121, -298, -139}
, {-258, -9, 188}
, {-29, 64, 55}
, {-51, -151, -95}
, {12, -29, -22}
, {84, 83, -176}
, {-82, -152, -139}
, {-103, -18, -203}
, {-12, 211, 128}
, {49, -99, 112}
}
, {{-296, -354, -199}
, {359, 133, -168}
, {-44, 40, 21}
, {24, 85, -34}
, {-229, -239, -95}
, {72, -198, -98}
, {67, -1, -129}
, {83, 497, 204}
, {15, 58, -134}
, {178, 27, -126}
, {-142, 83, 70}
, {132, -132, 174}
, {72, -6, -59}
, {-25, -73, 49}
, {93, 30, 498}
, {188, -472, -388}
}
, {{-191, -227, -224}
, {134, -260, -16}
, {16, 64, -96}
, {68, -25, -158}
, {-212, 0, -8}
, {37, 73, -129}
, {-19, -210, 105}
, {90, -368, 218}
, {-151, -230, -112}
, {-18, 121, -60}
, {-134, -257, -22}
, {-71, -166, 186}
, {-19, 234, -58}
, {-165, 308, 23}
, {-228, -140, 79}
, {60, 147, -63}
}
, {{-132, -171, 45}
, {1, 23, 170}
, {3, 31, -67}
, {-89, -9, -47}
, {-32, -62, 37}
, {89, 34, 159}
, {-188, -89, 1}
, {30, 137, 20}
, {175, -9, -90}
, {36, 41, -139}
, {-142, 116, 164}
, {129, 5, -108}
, {-302, 14, -74}
, {-215, -203, -267}
, {77, -7, 35}
, {-90, -83, 193}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE