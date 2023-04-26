/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 96
#define FC_UNITS 3


const int16_t dense_19_bias[FC_UNITS] = {106, -166, 52}
;

const int16_t dense_19_kernel[FC_UNITS][INPUT_SAMPLES] = {{-155, 147, 174, 119, -16, -2, 228, -46, 129, -188, -45, -122, 111, -81, -218, -138, -29, 119, 172, 24, -78, -130, 133, -72, -73, 5, -128, -66, 48, -9, 76, -79, 107, -138, 8, 107, -25, 112, 123, -150, 220, 283, 39, -168, -111, -264, -124, -232, 31, -10, -231, 10, 17, 132, -159, -167, 160, -49, 66, -85, 18, 7, -6, 91, 64, -28, -160, -152, -147, -201, 274, -191, -319, -115, 346, 76, 38, 3, 44, -28, -167, 157, -63, -158, -136, -91, 26, 63, 86, 95, 39, 113, -89, 6, 60, -38}
, {192, -99, -66, 77, -116, -157, -286, -86, -600, 117, -70, 67, 116, 105, 84, 2, -24, 3, -58, 68, -68, -29, 66, -19, 46, 39, 69, 21, 92, -12, 124, -179, 64, -34, -22, -149, -107, 24, 1, -239, 146, -286, 3, -71, -98, -53, 253, -181, 18, -51, 91, -16, -23, -113, 250, 137, -170, 88, -30, -8, -29, -6, 35, 144, 82, -94, 211, -36, -375, 192, -83, 231, 273, 253, -218, -65, -20, -6, 60, -83, -23, 55, 116, -30, 125, -78, 59, 11, -139, 77, 138, -114, 37, -15, 76, -68}
, {-131, 5, -118, 53, -82, 167, 36, 281, 277, 66, 130, 80, -154, 2, -127, 0, 51, -137, -204, 213, 464, 163, -66, -208, 71, 74, 307, 79, 62, 14, -280, 272, -64, 123, -59, -31, -56, 51, -26, 157, -187, -6, 2, 100, 231, 616, 119, 429, 43, 161, 104, -156, 37, -35, -218, 2, -70, -22, -74, 138, 69, -61, -81, 28, -103, 162, 101, 98, 346, 66, -599, -168, 167, -186, 25, -55, -84, 222, -117, -89, 71, -293, 139, -55, -40, 77, -156, 47, -62, 53, -61, -29, 79, 73, -23, -76}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS