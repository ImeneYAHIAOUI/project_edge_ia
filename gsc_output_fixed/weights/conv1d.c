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
#define CONV_KERNEL_SIZE  30


const int16_t conv1d_bias[CONV_FILTERS] = {0, -21, -129, -65, -247, -211, -446, -299}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-533, -115, 194, 61, -10, 21, -78, -60, 84, -9, 32, -60, 33, 259, 315, 15, -63, 197, 79, -59, 75, -39, -165, 114, 52, -29, 67, -138, -346, -309}
, {-426, -207, 121, 44, -66, -121, 22, 133, 190, 37, 74, 62, 14, 48, 179, 15, -193, 98, 118, 67, -51, -247, -284, -8, 90, -7, 111, 94, -89, -226}
}
, {{-72, -97, 91, 115, 217, 221, 76, 137, 225, 163, 204, 198, 35, 154, 206, 14, -57, 29, -79, -224, -112, -75, -12, 36, 27, -66, -101, -223, -56, 96}
, {-137, -117, -64, 156, 167, 295, 245, 5, 78, 57, 71, -73, 6, 73, 62, -160, -257, -113, -243, -263, -100, 6, -26, 1, 112, 9, -187, -245, -128, 57}
}
, {{-225, -158, -156, -29, 12, -85, 10, -88, -159, -111, -197, -81, -93, -41, 23, -56, -43, -14, -87, -35, -115, -64, -1, -117, -79, -81, -150, -48, -119, -167}
, {-157, -213, -115, -50, -64, 40, 0, -97, -81, -152, -131, -103, -70, -17, 31, -17, -138, -178, -151, -116, -29, -66, -91, -140, -225, -212, -107, -63, -92, -158}
}
, {{185, 183, 145, 118, 193, 245, 196, 173, 179, 145, 80, -7, 2, 36, -31, 84, 6, 98, 66, 80, 106, 242, 149, 257, 96, 144, 22, 23, 20, 64}
, {216, 131, 157, 129, 118, 196, 137, 153, 89, 38, -4, 101, 13, 115, 76, -4, 7, 62, 96, 147, 186, 220, 119, 154, 155, 155, 95, 56, 80, 169}
}
, {{-70, 53, 131, 72, 57, 66, 17, 20, -53, -1, -66, 30, 44, 24, 34, 24, 12, 62, 43, -55, -24, 22, 64, 40, 67, 62, 87, 19, 39, -3}
, {123, 134, 3, -7, -62, -179, -87, -110, 90, 47, 147, 89, 29, -23, -129, -157, -155, -100, -21, 51, 84, 75, 16, -26, -75, -90, -76, -51, 23, -4}
}
, {{275, 10, -22, -103, 13, 101, 176, 237, 280, 167, 190, 160, 93, -12, -147, -196, -175, -80, -77, 57, 75, 159, 215, 198, 179, 181, 56, 136, -27, 20}
, {-89, -209, -302, -150, -99, -46, -3, 11, -25, 77, 73, 54, -127, -174, -227, -236, -126, -77, -75, 66, 106, 89, -55, 25, -19, 1, 9, -125, -181, -221}
}
, {{80, 90, 89, -7, -56, -133, -112, -78, -76, -104, 24, 43, 51, 112, 33, -35, -113, -192, -139, -48, -59, 68, 86, 72, 68, 50, -41, -23, -152, -199}
, {88, 62, 99, -7, 9, -65, -57, -105, -40, 80, 12, 140, 72, 91, 85, 51, -34, -25, -39, -87, -31, 30, 92, 187, 30, 87, -64, -62, -112, -158}
}
, {{-24, -71, -10, -33, -46, 66, 70, 0, 73, 89, 135, 68, 49, 111, -21, 112, 132, 129, 16, 90, 102, -69, -34, 18, -125, 16, 76, 45, 152, 86}
, {-26, 70, -5, 55, 64, 0, -46, 55, 85, 110, 84, 171, 85, 98, 63, -32, -89, 41, 56, 4, 83, 1, 15, 87, -11, -81, 7, -88, -69, -181}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE