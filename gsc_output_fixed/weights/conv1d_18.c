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


const int16_t conv1d_18_bias[CONV_FILTERS] = {-52, -58, 60, 84, -56, -61, -86, -54, -52, -29, 313, 75, -120, -62, -81, -170}
;

const int16_t conv1d_18_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-38, -7, -164}
, {-111, -13, 53}
, {45, -150, -6}
, {-14, -90, -72}
, {-34, 16, -48}
, {39, -24, -51}
, {-5, -57, -3}
, {-90, 12, -13}
}
, {{18, -114, -62}
, {-185, -79, 104}
, {-177, 63, -176}
, {61, 50, -193}
, {-40, -154, -109}
, {116, 77, -89}
, {18, -35, -15}
, {-22, -179, -1}
}
, {{11, 22, 10}
, {-3, -178, -4}
, {104, -55, 3}
, {-109, -195, -54}
, {104, 17, -109}
, {116, -233, 21}
, {-185, -151, -171}
, {-73, -80, -61}
}
, {{-6, 11, 3}
, {15, -10, -129}
, {-130, -15, 91}
, {54, 18, -202}
, {88, -109, -60}
, {-162, 74, -202}
, {-45, -100, -82}
, {-37, 69, 74}
}
, {{77, 112, -26}
, {-65, 72, -158}
, {84, 13, 99}
, {-103, -111, 9}
, {-4, -52, -63}
, {84, -35, -105}
, {15, 90, -37}
, {-125, 84, 12}
}
, {{0, -138, -141}
, {-122, 48, 25}
, {24, -31, -114}
, {-118, -20, -204}
, {106, -49, 23}
, {-19, -6, 12}
, {-135, 122, -6}
, {52, -67, -92}
}
, {{103, 136, -29}
, {122, -93, -61}
, {-87, 112, -69}
, {-62, 19, -218}
, {-101, -5, -119}
, {-64, 60, -23}
, {-69, 120, -66}
, {-66, -80, -107}
}
, {{-212, -86, -46}
, {-15, -50, -32}
, {-25, 80, 30}
, {-118, -168, 28}
, {-146, -7, -132}
, {4, -202, -147}
, {-89, 11, -11}
, {-54, -142, -130}
}
, {{-58, -75, 44}
, {-28, -2, -2}
, {-178, 32, 13}
, {-91, -214, -5}
, {0, 78, -163}
, {-173, -165, -190}
, {79, -180, 33}
, {-211, -106, -179}
}
, {{33, -132, -72}
, {-93, 55, -155}
, {3, 44, 62}
, {-105, -138, -143}
, {10, -88, -40}
, {-106, -158, -160}
, {-2, -169, 3}
, {77, -14, 19}
}
, {{52, 20, -11}
, {-39, 6, -194}
, {-180, -174, -183}
, {-210, 18, 37}
, {70, 63, -124}
, {-26, -129, -192}
, {23, -129, -38}
, {-6, 71, -12}
}
, {{1, 4, -108}
, {-13, -100, -34}
, {-111, -72, -57}
, {-185, -180, 89}
, {-35, -179, 34}
, {-51, 58, -72}
, {-169, -167, -130}
, {59, 41, 56}
}
, {{102, -87, -54}
, {-28, 50, -97}
, {-47, -120, -162}
, {44, -158, -129}
, {4, -144, 1}
, {-206, -23, 73}
, {-86, -89, -6}
, {7, -66, 96}
}
, {{34, -65, -102}
, {-88, -101, 33}
, {71, -175, 28}
, {13, -90, -6}
, {-57, 126, 63}
, {-97, -175, 58}
, {37, -102, -54}
, {93, -122, -22}
}
, {{51, -145, -112}
, {72, 40, -106}
, {54, 14, -79}
, {67, 49, -56}
, {-19, -140, 5}
, {-107, -131, -27}
, {-123, 16, -76}
, {122, -49, -104}
}
, {{50, -48, -15}
, {-126, 20, 103}
, {48, -61, -34}
, {140, -77, 52}
, {12, 34, -153}
, {-20, -149, 122}
, {-13, 144, 22}
, {-38, 11, 70}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE