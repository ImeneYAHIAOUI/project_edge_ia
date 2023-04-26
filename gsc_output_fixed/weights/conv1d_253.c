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


const int16_t conv1d_253_bias[CONV_FILTERS] = {17, -61, -125, 189, 9, -188, -10, 57, -79, -43, 66, 150, 43, 11, -53, -192}
;

const int16_t conv1d_253_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-369, 84, 194}
, {70, -22, -1}
, {70, -165, 23}
, {417, 141, 56}
, {-101, -24, -113}
, {-6, 41, -72}
, {-135, -11, -72}
, {-106, 19, -45}
, {-241, -269, 20}
, {141, 76, -35}
, {-87, -27, -120}
, {84, 11, -85}
, {-147, -73, -190}
, {36, 71, 56}
, {-44, -8, -52}
, {331, 178, -83}
}
, {{-74, -204, -65}
, {38, 75, 157}
, {174, 10, 63}
, {-171, -186, -252}
, {-75, -61, -113}
, {91, 37, -107}
, {81, -22, 68}
, {33, 106, -53}
, {-123, -43, 69}
, {113, 12, 72}
, {96, -46, -61}
, {-120, -5, -71}
, {30, -33, -112}
, {47, -86, -111}
, {-16, -140, -108}
, {-251, -41, 109}
}
, {{-29, 84, 159}
, {-18, -26, 7}
, {-188, 54, 65}
, {329, 267, 96}
, {125, 106, 171}
, {83, 62, 63}
, {93, -115, 5}
, {0, -6, -5}
, {-164, -33, -11}
, {113, -93, 13}
, {91, -18, -4}
, {-55, 91, 97}
, {21, 75, 69}
, {-3, -104, 16}
, {-86, -103, -29}
, {-29, -105, -75}
}
, {{38, 92, -71}
, {-159, -357, -207}
, {117, -254, 121}
, {-65, -31, 212}
, {-291, -112, -44}
, {40, 48, 30}
, {-431, -326, -22}
, {-148, -172, -280}
, {-248, -117, -130}
, {-146, -92, -144}
, {0, -32, 142}
, {-1, -42, -7}
, {-427, -157, 24}
, {267, 113, -135}
, {15, 222, 137}
, {170, 101, 31}
}
, {{-86, -19, 36}
, {-215, -78, -93}
, {-156, -136, -201}
, {-116, -45, 26}
, {143, 4, -236}
, {81, 53, 92}
, {-152, -121, -54}
, {-84, 22, 74}
, {-151, -84, 119}
, {-146, 59, 65}
, {82, 28, 106}
, {-52, 115, -113}
, {23, 98, 55}
, {23, 150, 4}
, {220, -48, 0}
, {-105, 244, -172}
}
, {{-19, -7, -138}
, {-100, -76, -224}
, {6, -259, -88}
, {-159, 16, -42}
, {-226, -135, -290}
, {26, 54, -88}
, {-83, -185, -163}
, {-222, -48, -142}
, {-126, 49, -290}
, {-173, -196, -8}
, {-52, 19, -124}
, {-99, 111, -98}
, {20, 0, -71}
, {-148, 161, -34}
, {111, -132, -19}
, {188, 265, -49}
}
, {{-122, -74, -61}
, {129, -105, -135}
, {110, -87, 0}
, {-195, 53, -79}
, {-180, -11, -132}
, {-41, -68, -108}
, {-18, -13, -108}
, {-4, 17, -105}
, {-191, -244, 33}
, {-163, -7, -29}
, {-16, -68, -127}
, {5, 46, -51}
, {-144, -55, -95}
, {45, -37, 95}
, {-157, 57, 15}
, {-202, 15, 15}
}
, {{83, -19, 81}
, {-144, 102, -70}
, {-35, -283, -318}
, {37, 7, -123}
, {-106, -248, 127}
, {41, 109, 22}
, {0, -75, -194}
, {-94, 23, -86}
, {174, -10, 55}
, {-21, 24, -246}
, {104, 48, -119}
, {94, 84, -45}
, {190, -207, -128}
, {150, 110, -31}
, {80, 154, -229}
, {-77, 44, 169}
}
, {{-54, -238, -43}
, {-21, -51, -115}
, {96, 108, -2}
, {94, -258, -67}
, {-146, -120, -1}
, {4, 75, -118}
, {9, -49, -15}
, {90, 127, 34}
, {-166, -69, -117}
, {-183, -40, 45}
, {88, -1, 112}
, {89, -34, 10}
, {-64, -122, 92}
, {74, 55, -68}
, {-116, -135, -125}
, {-142, -236, -26}
}
, {{-182, -149, -33}
, {-4, -85, -53}
, {-120, 27, -23}
, {-78, 48, -123}
, {-4, -194, -31}
, {46, 74, -47}
, {11, -51, -112}
, {-93, -79, -130}
, {16, 53, 10}
, {-161, 43, -14}
, {75, -123, 23}
, {79, -137, -26}
, {-26, -57, -26}
, {-95, -24, 95}
, {-5, -19, -119}
, {-74, -149, -88}
}
, {{-106, 180, -101}
, {97, -70, -189}
, {49, -26, 66}
, {325, 93, 47}
, {235, 168, 320}
, {-119, -14, -123}
, {-230, -278, -233}
, {25, -95, -4}
, {-119, 66, -67}
, {82, 63, -89}
, {16, -53, 47}
, {46, 60, 30}
, {-353, -136, -76}
, {291, -14, -84}
, {230, -258, 61}
, {-119, -37, -184}
}
, {{-36, -11, -64}
, {-283, -70, 35}
, {-120, 104, 160}
, {42, -117, -8}
, {74, -25, -190}
, {57, 12, 55}
, {163, -56, -205}
, {-25, -109, -4}
, {-258, -121, 277}
, {-109, -76, 48}
, {0, -38, 90}
, {-69, -87, -63}
, {-181, 85, 101}
, {98, 9, -28}
, {-64, -223, 57}
, {252, -38, 17}
}
, {{325, 119, -47}
, {-126, -75, -182}
, {1, -147, -186}
, {-254, 15, 248}
, {152, -77, -67}
, {66, 34, -10}
, {-107, -14, 117}
, {-196, -43, -68}
, {-50, 127, -35}
, {69, -131, -221}
, {46, -36, 11}
, {101, -130, -163}
, {73, -211, -280}
, {64, -84, 15}
, {373, -120, 104}
, {277, -97, -200}
}
, {{-252, -103, 14}
, {-142, -119, -30}
, {36, -138, -106}
, {258, 132, -164}
, {-96, 75, -22}
, {-95, 57, -62}
, {41, 127, 124}
, {-24, -105, -75}
, {-9, 79, 15}
, {10, -112, 33}
, {86, 45, 59}
, {3, 104, -50}
, {-143, -69, 79}
, {69, 50, -23}
, {66, -24, -88}
, {-90, -104, 44}
}
, {{23, -42, 88}
, {-139, 14, 295}
, {52, -400, -120}
, {221, 150, -12}
, {-132, 98, 3}
, {-32, -45, 39}
, {-266, -181, 239}
, {-151, 4, -153}
, {-64, 134, 160}
, {81, 31, -135}
, {55, 80, -116}
, {-32, 27, 87}
, {123, -18, -239}
, {-37, -100, -85}
, {-256, -202, -344}
, {176, 8, 77}
}
, {{-145, -176, -70}
, {74, -259, -142}
, {241, -144, 103}
, {-386, 23, -188}
, {-160, 38, 130}
, {106, 58, 132}
, {4, 58, -95}
, {-110, -12, 13}
, {-205, -21, 118}
, {-102, -19, -128}
, {-39, -65, 66}
, {7, -117, 26}
, {160, -141, 227}
, {35, -30, -74}
, {70, -280, -88}
, {-59, -180, -92}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE