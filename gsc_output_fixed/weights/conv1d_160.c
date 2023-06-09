/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    2
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  40


const int16_t conv1d_160_bias[CONV_FILTERS] = {-165, -18, -27, -47, -39, -196, -269, -217}
;

const int16_t conv1d_160_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-9, 9, -66, -89, 47, -72, -36, 15, 47, 12, -9, -26, 75, -10, -14, 28, -24, 15, 21, 96, -3, 30, 3, -95, -66, 19, -81, 0, 81, 26, 42, -18, -44, -16, -19, -2, -7, -51, -6, -35}
, {-23, -80, -119, -235, -217, -68, 31, 41, 1, 23, 66, 37, -39, -127, -235, -157, -24, 82, 186, 130, -1, -38, -22, -21, -88, -41, -3, 1, 119, 118, 102, -48, -176, -159, -197, -184, -210, -70, -59, -4}
}
, {{-3, -37, -12, -37, -40, -12, -34, 11, 4, 122, 16, -59, -111, -153, -209, -303, -242, -130, -218, -93, -22, -15, 84, 102, 32, 11, -115, -9, 84, 99, 59, 48, 68, 143, 166, 31, 32, -95, -53, -46}
, {-80, 165, 129, 40, 123, 183, 199, 77, 77, -46, -139, -99, -73, 50, 110, -37, -105, -47, -206, -184, 3, -3, -13, 50, 7, 242, 199, 236, 178, -70, -94, -8, -41, 86, -34, 21, 4, 105, 223, 87}
}
, {{-38, 45, 10, 45, 47, 77, 86, 81, 37, 29, 55, 98, 146, 93, 60, 78, 103, 30, 30, -7, -56, -34, -8, -17, 27, 18, 20, 57, 26, 31, 24, 83, 107, 117, 35, 104, 11, 64, -19, 3}
, {59, 26, 90, 94, 70, 70, 119, 130, -22, 86, 56, 105, 97, 171, 157, 100, 262, 170, 227, 161, 74, 113, 2, 76, 46, 105, 68, 113, 108, 73, 117, 44, 158, 142, 119, 137, 96, 86, 68, 82}
}
, {{153, 207, 99, -93, -159, -137, -18, -13, 13, -50, 31, 120, 120, 56, 68, -52, -60, -52, -67, -155, -158, -134, -30, 77, 64, 102, 119, 46, 13, 22, -76, -84, -52, -32, -42, -37, 40, 52, 123, 148}
, {127, 37, -8, 11, -48, 18, -63, -92, -69, -109, 90, 33, 40, 111, 66, 49, 38, -221, -266, -345, -154, -82, 63, 67, 56, 28, 22, 86, -3, -64, -127, -159, -125, -125, 77, 165, 160, -5, -19, -63}
}
, {{-109, -45, -23, -100, -65, -144, -42, -63, 40, -40, -71, -15, -88, -50, -99, -51, 52, 25, 41, -4, -14, -112, -91, -123, -101, -101, -91, -66, -15, -64, -79, 17, -26, -50, -96, -95, -55, -3, 76, 30}
, {-31, -69, -134, -152, -159, -176, -131, -98, -30, -38, -78, -36, -127, -27, -70, -131, -38, -94, -108, -154, -168, -107, -72, -43, -70, -142, -141, -111, -74, 0, 7, -33, 5, 18, 17, 120, 67, 132, 138, 113}
}
, {{17, 98, 89, 8, -28, -9, -82, 36, -72, -6, -83, -173, -128, 34, 45, -33, -117, -137, -62, 25, 34, 98, -16, 48, -24, -10, -19, -13, -74, 0, -9, -24, 47, 49, -5, -65, -104, -69, 47, 43}
, {8, -136, -192, -304, -125, 33, 111, 78, -113, -275, -218, -97, 44, 190, 91, -138, -62, -136, -130, -126, -87, -179, -186, 1, 81, 31, 43, -200, -189, -275, -137, -42, -158, -129, -170, -266, -186, -43, -127, -118}
}
, {{-93, -99, -15, -86, -16, -21, 18, -4, -27, -108, -52, -7, -2, -82, -42, -42, -169, -118, -169, -132, -53, -27, -65, -111, -98, -126, -48, -67, -53, -98, -117, -72, -60, -42, -163, -96, -77, -88, -130, -191}
, {9, 51, -89, -84, -169, -85, -53, -130, -3, 6, -57, -52, -6, 28, -54, -74, -9, -140, -6, -16, -64, -34, -54, -24, -130, -132, -81, -134, -151, -54, -46, 115, 50, 43, -59, -90, -93, -176, -131, -100}
}
, {{132, 71, 2, -7, -44, 71, -6, 21, 13, 31, 75, 24, 36, 25, 11, 23, -93, -7, -33, -64, 19, -32, 4, 66, 90, 124, 70, 92, 52, -13, 75, -31, -21, -117, -51, -28, 0, 86, 47, 165}
, {45, -19, -68, 46, 63, 89, 59, 27, 26, 55, -91, -78, -45, -101, -123, -69, -72, -52, -70, -30, 37, 26, 45, 23, 72, -7, 4, -38, -66, -94, -104, -64, -85, -96, -16, -8, -108, -39, -29, 26}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE