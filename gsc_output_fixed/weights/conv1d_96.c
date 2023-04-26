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


const int16_t conv1d_96_bias[CONV_FILTERS] = {-43, -38, 40, -53, -39, -38, -50, -43, -49, 37, -26, -48, -11, -18, -48, -22}
;

const int16_t conv1d_96_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{74, -140, -64}
, {-154, -129, 41}
, {50, -19, 36}
, {61, 42, -125}
, {34, -102, -34}
, {-147, -25, 34}
, {50, 46, -41}
, {-88, 32, -57}
, {-131, -53, -163}
, {66, -170, -60}
, {-39, -57, 21}
, {-156, -57, -19}
, {-42, 49, -81}
, {-15, -94, -10}
, {20, -171, 80}
, {-80, -28, 20}
}
, {{57, 13, 80}
, {-142, -58, -54}
, {-107, -40, -82}
, {-5, -147, 88}
, {-143, -15, -149}
, {-30, -85, -33}
, {59, 24, -60}
, {-150, -82, 86}
, {-105, -14, 92}
, {30, 14, -47}
, {8, -5, 87}
, {93, 85, -16}
, {49, 79, -30}
, {-114, -152, -124}
, {-72, 36, -127}
, {-81, -99, 11}
}
, {{-88, -110, 6}
, {67, -150, 35}
, {-72, 0, -131}
, {73, 93, -21}
, {23, 74, -126}
, {30, -135, -103}
, {119, -93, -38}
, {-7, -11, -46}
, {12, 119, -13}
, {95, 71, -153}
, {184, 41, 65}
, {50, -6, 41}
, {60, -48, 4}
, {-49, 47, 41}
, {115, 3, 18}
, {134, -14, -114}
}
, {{43, -58, -25}
, {-14, 156, -51}
, {-23, -146, 109}
, {-81, 127, 62}
, {-69, 48, 17}
, {81, -20, -19}
, {20, -52, 24}
, {21, 20, 36}
, {-29, 75, 85}
, {-211, -45, 55}
, {49, -14, -159}
, {-142, -102, -115}
, {-20, 112, -123}
, {30, 92, -18}
, {-98, -89, -155}
, {-105, -2, -114}
}
, {{-156, -58, -15}
, {-67, -22, -136}
, {17, -98, -146}
, {-46, -142, -33}
, {-137, -27, 0}
, {-3, -64, 72}
, {-50, 67, -54}
, {-9, 22, -34}
, {-48, -145, -85}
, {38, 73, -6}
, {-84, 110, -89}
, {-120, -128, -15}
, {31, -19, -38}
, {-96, 92, -68}
, {68, -1, 64}
, {-188, -86, 84}
}
, {{-83, 45, 50}
, {68, 19, -62}
, {44, -36, 24}
, {1, -10, -84}
, {47, 18, -125}
, {-27, 66, -113}
, {26, -137, 48}
, {-118, -106, -122}
, {-84, 67, 81}
, {31, 9, -123}
, {-50, 31, 61}
, {-8, -66, 2}
, {-119, -90, -43}
, {87, 122, 84}
, {-123, -12, 73}
, {-37, -106, 32}
}
, {{48, -66, -70}
, {-31, -158, -46}
, {-124, -35, -155}
, {-105, -176, -138}
, {19, 42, -32}
, {60, -43, 25}
, {-155, 17, -153}
, {-152, -159, 12}
, {-158, -134, -97}
, {-157, -51, -14}
, {-30, 9, -83}
, {-78, 38, -10}
, {-37, -55, -47}
, {67, -145, 45}
, {20, -69, -80}
, {-27, -36, -73}
}
, {{32, -122, 104}
, {-118, -5, -64}
, {41, 2, -104}
, {-88, -153, 63}
, {-97, 25, -78}
, {-35, 7, -142}
, {-89, -5, -33}
, {13, 22, -145}
, {-161, 18, 60}
, {50, -81, -109}
, {61, 67, -14}
, {-121, -27, 84}
, {-115, 32, 72}
, {10, -64, -38}
, {-119, -78, -130}
, {-175, -151, 17}
}
, {{-88, 20, 24}
, {2, -27, 42}
, {-149, 48, 11}
, {14, 7, -35}
, {-5, -80, -164}
, {62, 61, -121}
, {-165, -55, 12}
, {44, 15, -78}
, {-112, -38, 61}
, {-79, -61, -20}
, {-44, -42, -112}
, {-121, 45, 72}
, {65, -123, -56}
, {-106, 58, 68}
, {-10, -125, -19}
, {-4, 37, -135}
}
, {{83, 150, 23}
, {-90, -45, -111}
, {37, -127, 92}
, {37, 96, 174}
, {-51, 3, 78}
, {-145, -56, -138}
, {-126, -117, -40}
, {80, 122, 39}
, {98, 20, -10}
, {-99, -164, -96}
, {-106, -66, -120}
, {157, 82, 133}
, {-26, 58, -129}
, {109, -106, 80}
, {-17, -13, -10}
, {-61, -134, 31}
}
, {{-47, 1, 84}
, {-157, -25, 58}
, {5, -82, -124}
, {67, 81, -58}
, {-62, -1, -107}
, {-34, 38, -99}
, {-50, -115, 11}
, {-52, 37, -62}
, {-66, -153, -152}
, {-26, 51, -149}
, {-69, -97, 82}
, {-35, -12, -14}
, {-108, 60, 82}
, {-8, -123, -100}
, {15, 22, -120}
, {-19, 95, -115}
}
, {{-109, -32, 56}
, {-68, -153, -161}
, {-127, -176, 59}
, {78, -123, -34}
, {-63, 26, -123}
, {-82, -134, -10}
, {5, 31, -59}
, {62, 55, 17}
, {-166, 6, 25}
, {-140, -119, 43}
, {62, 51, -83}
, {46, -47, 31}
, {6, -23, 84}
, {-124, 21, 68}
, {51, -13, -59}
, {-165, -118, -4}
}
, {{-5, -2, -4}
, {-23, 75, 10}
, {-29, 51, 38}
, {-14, 71, -130}
, {-60, -71, -4}
, {81, -81, -109}
, {-119, 51, 19}
, {-36, -114, -112}
, {40, 54, -37}
, {6, -19, -83}
, {-55, 54, 62}
, {65, -20, 28}
, {-73, -19, 28}
, {-103, -26, -94}
, {-38, -122, 21}
, {59, -115, 44}
}
, {{148, -7, -5}
, {-13, -75, -124}
, {85, 179, -47}
, {142, 69, 61}
, {52, -96, 99}
, {5, 84, 17}
, {-88, 19, 58}
, {-39, 4, 155}
, {123, 41, 64}
, {-4, 49, -56}
, {-1, -82, -66}
, {159, 95, 116}
, {25, 87, -85}
, {-5, 23, 12}
, {91, -52, -45}
, {-36, 11, 137}
}
, {{-29, -132, -94}
, {72, -32, 39}
, {-50, -125, 40}
, {-77, -48, -126}
, {78, -132, 59}
, {-129, -149, 57}
, {-93, 16, -122}
, {23, -140, -108}
, {-72, 54, -43}
, {-166, -65, -33}
, {-128, 29, 67}
, {49, -41, -22}
, {30, -97, 48}
, {-152, -22, -138}
, {-136, -1, 7}
, {56, -45, 62}
}
, {{78, -90, -103}
, {37, -137, 37}
, {-74, -94, -122}
, {34, 107, 38}
, {28, 80, 59}
, {12, 83, -23}
, {-5, -63, 115}
, {-28, 82, -123}
, {-32, 94, -40}
, {-20, -92, -25}
, {-54, 39, -128}
, {-43, -57, 11}
, {-69, -32, -39}
, {-22, -25, 41}
, {-79, -5, 12}
, {-87, 31, 61}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE