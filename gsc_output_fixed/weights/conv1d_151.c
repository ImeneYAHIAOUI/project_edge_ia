/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_151_bias[CONV_FILTERS] = {71, 4, -121, -123, -79, -15, -159, -23, -37, 97, 117, 17, -122, -2, -78, 118, 84, -150, 34, 83, -153, -108, 41, -9, -152, -184, -78, 19, -98, -104, -29, 83}
;

const int16_t conv1d_151_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-104, 8, -83}
, {10, 11, 54}
, {-1, 27, 39}
, {-21, -82, -68}
, {1, -77, -117}
, {-26, -17, 5}
, {-1, -37, -48}
, {53, -91, -17}
, {66, -90, -109}
, {4, -25, 20}
, {26, -26, -51}
, {-3, -60, -65}
, {-37, -103, 10}
, {-14, -22, -41}
, {-5, 12, -118}
, {-51, -64, -103}
}
, {{11, 37, 12}
, {46, -118, -77}
, {73, -17, -80}
, {-10, 127, 18}
, {32, -55, -90}
, {-18, 63, 14}
, {27, -95, -54}
, {-86, -123, -15}
, {23, -107, -61}
, {67, -9, 65}
, {-57, -70, 3}
, {-106, -66, -107}
, {-47, 18, 62}
, {-125, -3, -91}
, {-89, 49, -135}
, {-42, -55, -24}
}
, {{-57, 20, -65}
, {-15, 1, -10}
, {72, -73, 16}
, {95, 6, 0}
, {86, -29, -1}
, {61, 19, 37}
, {-30, -51, -8}
, {45, -118, -96}
, {14, -34, 73}
, {-91, 9, -95}
, {-70, 87, -35}
, {87, -41, -106}
, {60, -16, -39}
, {68, 63, 3}
, {89, -87, 76}
, {-113, 112, -25}
}
, {{4, 56, 74}
, {-48, 80, 42}
, {60, 51, 81}
, {86, 47, 1}
, {45, -17, -90}
, {78, -101, 90}
, {-95, 68, -88}
, {-20, -19, 30}
, {-50, 89, 83}
, {77, -91, 81}
, {-49, -57, 53}
, {62, -91, -4}
, {8, 68, 32}
, {-96, 96, 92}
, {-43, 104, 26}
, {-36, 71, 22}
}
, {{95, 45, -63}
, {84, -98, -23}
, {78, 29, -28}
, {-63, -80, 81}
, {13, 74, 31}
, {-19, -52, -71}
, {-34, -11, -27}
, {60, -87, 115}
, {-13, -122, -44}
, {-112, -116, -73}
, {-80, 40, -9}
, {-6, -90, -85}
, {85, 55, -29}
, {91, -47, 87}
, {30, -106, -80}
, {-80, 13, 7}
}
, {{-127, -116, -8}
, {14, 23, 16}
, {-110, -104, 3}
, {-78, -133, 59}
, {-67, -11, -61}
, {-69, 46, 16}
, {-52, 73, 14}
, {-63, 55, 55}
, {-27, -104, -75}
, {-105, -55, 84}
, {-12, -68, 26}
, {31, 34, 27}
, {-105, -72, 98}
, {-112, -17, 19}
, {-96, 6, 61}
, {14, -86, -56}
}
, {{-11, 23, -84}
, {89, -47, -6}
, {98, 117, 21}
, {37, 124, -22}
, {33, -15, 5}
, {107, 7, 37}
, {-25, -62, -29}
, {40, 118, 14}
, {-57, -90, 56}
, {10, 23, -85}
, {-57, -1, 40}
, {40, -28, 89}
, {27, -82, -19}
, {87, -95, -82}
, {-106, 33, -16}
, {-88, 37, -19}
}
, {{68, -9, -9}
, {-143, -1, -32}
, {-71, 46, -16}
, {30, 62, -12}
, {21, -34, -79}
, {70, -2, 23}
, {-72, -22, -44}
, {-81, -92, 32}
, {-47, -9, -118}
, {-13, -109, 24}
, {-105, -16, -36}
, {-52, 20, 98}
, {41, -49, -61}
, {77, 92, 20}
, {58, -118, 55}
, {74, 23, -74}
}
, {{-13, -13, 38}
, {43, -99, 77}
, {31, -62, 43}
, {-47, -49, 15}
, {20, 62, -79}
, {-21, -59, 81}
, {-30, 18, 0}
, {76, 11, 34}
, {59, -18, -59}
, {31, 78, -23}
, {72, -106, 64}
, {-82, -52, -56}
, {10, 13, -66}
, {92, -125, -90}
, {-116, 61, 18}
, {5, -107, 58}
}
, {{97, -43, -57}
, {-70, 9, 53}
, {-94, -70, 21}
, {26, 50, -26}
, {51, 1, -74}
, {56, -78, 48}
, {-29, -11, -93}
, {-29, 10, -35}
, {-86, 19, 45}
, {-37, 5, -28}
, {29, -13, -46}
, {8, -37, -103}
, {77, 36, 73}
, {41, -24, 79}
, {0, 52, 25}
, {63, 26, 1}
}
, {{7, -74, 20}
, {-94, -33, -115}
, {-23, -55, 53}
, {94, -91, -22}
, {-33, 7, -113}
, {-24, 3, 19}
, {-36, 37, -9}
, {67, -26, 10}
, {-129, -24, 35}
, {-38, 27, 63}
, {7, -109, 29}
, {-13, 66, -17}
, {-43, 14, -116}
, {25, -61, 30}
, {-119, -8, -76}
, {54, 94, -1}
}
, {{-14, -48, -69}
, {-20, -99, -127}
, {72, -22, -77}
, {-119, -47, -153}
, {-63, -123, -10}
, {60, 78, -75}
, {-42, -29, 72}
, {-29, 46, -5}
, {20, -131, -119}
, {-65, 40, -102}
, {-102, 78, -77}
, {-10, 21, -13}
, {-80, 6, -107}
, {20, 33, -124}
, {-39, -129, 27}
, {36, 29, 45}
}
, {{34, -61, -123}
, {-17, -83, -49}
, {-141, 21, -69}
, {59, 1, -16}
, {-59, 53, -98}
, {44, 21, -65}
, {36, -81, 20}
, {56, -83, -55}
, {-103, -92, -81}
, {-21, -33, -59}
, {77, 51, -26}
, {-108, 37, 72}
, {58, 68, -79}
, {-67, 39, -41}
, {47, 18, -2}
, {-123, -73, -29}
}
, {{76, 33, -96}
, {-46, -105, -110}
, {-54, 12, 32}
, {62, -118, -115}
, {49, -84, 5}
, {-124, 52, 69}
, {35, -98, 68}
, {-113, -100, 44}
, {-10, -8, -94}
, {55, -35, 64}
, {12, -68, -106}
, {-90, -40, 57}
, {-58, -38, -84}
, {38, 55, 16}
, {-58, -49, -3}
, {-87, 11, 45}
}
, {{-71, -43, 84}
, {-48, -39, 35}
, {-101, -23, 68}
, {4, 35, -5}
, {-4, -23, -107}
, {47, 10, -83}
, {42, -109, -28}
, {22, -36, -96}
, {-39, -98, -29}
, {37, -58, -53}
, {-68, -100, 33}
, {50, 25, -54}
, {-69, -80, -76}
, {-129, 0, 25}
, {39, 34, -75}
, {-68, -36, -70}
}
, {{26, 60, -82}
, {95, -57, -10}
, {-24, 62, 20}
, {46, -45, -41}
, {44, 80, -82}
, {-95, -57, 34}
, {-107, -103, -13}
, {23, -12, 115}
, {-9, 38, 3}
, {-105, -59, 15}
, {-29, 44, 83}
, {59, -8, 3}
, {83, -88, -37}
, {-98, -80, 23}
, {61, -130, 0}
, {-37, -81, 90}
}
, {{76, -3, -54}
, {33, 35, -49}
, {-75, -115, -95}
, {-7, 72, 21}
, {49, 89, 60}
, {30, 37, -61}
, {-108, -30, -81}
, {4, -40, -124}
, {24, -4, -78}
, {85, 11, 88}
, {95, 85, -15}
, {68, 56, -5}
, {-112, -21, -40}
, {-89, -55, 24}
, {-86, -12, 32}
, {-15, 3, -5}
}
, {{80, -3, 63}
, {-24, 43, 52}
, {-48, 107, -38}
, {37, 86, 37}
, {-83, -18, 37}
, {-72, 68, -1}
, {-39, 23, -86}
, {10, 35, -40}
, {-58, 54, 32}
, {62, 32, -1}
, {15, -85, 82}
, {64, -22, -29}
, {50, -6, 3}
, {31, -8, -61}
, {-20, 28, 91}
, {-109, 28, -43}
}
, {{-126, -32, -2}
, {-42, 37, -68}
, {-13, 80, 65}
, {-7, 52, -116}
, {-96, 25, -82}
, {-8, -23, -39}
, {0, -61, -8}
, {-18, -92, -120}
, {-5, 34, -76}
, {-22, 16, -78}
, {-112, -59, -11}
, {51, -93, -51}
, {0, -102, -1}
, {3, -103, -41}
, {-13, -42, -14}
, {-18, -137, -59}
}
, {{-78, 72, -13}
, {-1, 96, -85}
, {-28, -124, 40}
, {74, 7, -47}
, {68, -53, -78}
, {62, 51, -97}
, {0, -27, -53}
, {99, -16, 66}
, {-110, 31, -12}
, {40, -20, 22}
, {6, -114, -118}
, {57, -62, -11}
, {29, 52, 40}
, {-10, 39, 17}
, {-70, -22, -18}
, {-101, 73, 82}
}
, {{-107, -59, -57}
, {9, 57, -87}
, {-38, -46, 99}
, {70, 121, 44}
, {54, -53, 62}
, {-17, 79, 16}
, {-107, -87, 52}
, {47, -5, 116}
, {67, 50, 65}
, {28, -87, -76}
, {-2, -119, -61}
, {91, -3, 85}
, {37, 12, -85}
, {0, 47, 86}
, {-127, -101, -18}
, {-98, -73, -65}
}
, {{47, 109, 91}
, {3, 39, 17}
, {70, 22, 9}
, {124, -114, 68}
, {-105, 71, 53}
, {-5, 39, 27}
, {43, 20, 68}
, {-50, -7, -84}
, {-48, 11, 53}
, {23, 88, -4}
, {-107, 63, 4}
, {11, 31, 110}
, {79, 106, -48}
, {111, 46, -50}
, {-102, 42, -20}
, {-6, 36, -2}
}
, {{39, 54, 2}
, {-99, 14, 109}
, {-54, -30, 61}
, {-117, 67, -8}
, {-45, -33, -60}
, {-55, 50, 11}
, {2, -15, -36}
, {-1, -54, -14}
, {89, 58, 4}
, {-101, -17, -52}
, {27, 62, 20}
, {126, 21, -30}
, {65, 76, -81}
, {-106, -62, -15}
, {40, -96, 14}
, {65, -55, -7}
}
, {{73, -73, 74}
, {5, -109, -119}
, {91, 37, -26}
, {94, 13, 33}
, {-67, -39, -66}
, {80, -68, 128}
, {-14, 34, 81}
, {55, -44, -14}
, {-119, -21, 38}
, {34, 36, -31}
, {-113, -41, 58}
, {-36, 92, 84}
, {-5, 39, -20}
, {-84, 12, -54}
, {-69, -107, 7}
, {56, -31, -36}
}
, {{-58, -72, 81}
, {73, -25, -73}
, {36, -31, 113}
, {-81, 37, -8}
, {-48, -93, 80}
, {128, 71, -59}
, {4, -61, 111}
, {24, 17, 64}
, {-78, 88, -76}
, {5, 79, 89}
, {-35, 78, 27}
, {-26, 38, -11}
, {-67, 85, 74}
, {106, 4, 24}
, {-63, -88, 24}
, {17, 21, -9}
}
, {{-63, -130, 11}
, {-61, 51, 7}
, {58, -98, -54}
, {-61, -54, 34}
, {63, -93, -8}
, {-16, 79, -53}
, {-74, -33, 86}
, {109, -157, 24}
, {-101, 3, 94}
, {-27, -114, -40}
, {51, -97, 27}
, {41, -58, -111}
, {-59, -48, -10}
, {65, 17, -7}
, {-44, -29, 28}
, {28, 18, -6}
}
, {{1, -1, -2}
, {-19, 59, -125}
, {-2, 74, -82}
, {14, 19, -39}
, {17, -85, -90}
, {15, 35, -70}
, {44, -35, -84}
, {24, -107, -66}
, {-29, -76, -37}
, {75, 49, -89}
, {31, -35, 13}
, {-26, -69, -16}
, {-56, 1, -49}
, {-78, -62, 58}
, {-52, 53, -102}
, {-44, -102, 9}
}
, {{-45, -1, 55}
, {-71, 83, 28}
, {-116, -15, -15}
, {-72, -90, -32}
, {23, -80, -75}
, {12, -87, -27}
, {8, 29, -19}
, {-21, -107, 46}
, {-73, -70, 15}
, {24, 13, 73}
, {-14, 20, 68}
, {19, -121, 38}
, {-98, 64, -92}
, {-39, 86, -32}
, {-48, 31, 74}
, {-61, 62, 70}
}
, {{-30, 83, -38}
, {-36, 5, 83}
, {51, 32, 26}
, {-50, 76, 142}
, {91, -59, -36}
, {2, -31, -18}
, {104, -45, -94}
, {-36, -9, 56}
, {63, 41, -72}
, {89, 1, 21}
, {83, -116, -49}
, {39, 17, 62}
, {-13, 2, 74}
, {65, 30, -80}
, {-67, -65, -20}
, {120, -122, 46}
}
, {{33, 52, 62}
, {88, 86, -113}
, {77, 69, -72}
, {-26, -78, -23}
, {-103, 86, -64}
, {-20, -60, 25}
, {-16, -53, -30}
, {38, -62, -115}
, {-14, 48, -54}
, {64, 53, -50}
, {67, -89, -58}
, {120, 128, 99}
, {58, 87, -66}
, {-34, 25, -37}
, {-3, 60, -76}
, {-48, -89, -100}
}
, {{-85, 16, 54}
, {-92, 88, 47}
, {-125, -51, 27}
, {-12, -39, -41}
, {63, 33, 31}
, {68, -80, -2}
, {-59, -72, -33}
, {67, -53, 21}
, {-22, 43, -59}
, {81, 18, 75}
, {21, 67, -41}
, {3, 31, 54}
, {-31, 32, 84}
, {-90, -79, -94}
, {9, -31, -52}
, {112, 88, -60}
}
, {{34, 94, -109}
, {78, 64, 97}
, {-52, 6, -125}
, {-28, 50, -3}
, {-98, -38, 13}
, {-34, -1, 80}
, {0, -114, 5}
, {-21, 67, -62}
, {-64, 27, -29}
, {-3, 70, 52}
, {82, -27, -78}
, {-33, 60, -57}
, {60, 0, -27}
, {-35, 19, -23}
, {46, -64, 68}
, {28, 41, 54}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE