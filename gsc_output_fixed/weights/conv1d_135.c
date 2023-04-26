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


const int16_t conv1d_135_bias[CONV_FILTERS] = {-13, -40, -51, -75, 3, -42, 9, -21, -51, 67, -27, -4, -39, -52, -25, 152, 81, -35, -55, -42, -35, -36, -61, -33, -50, -39, -28, -27, -14, 0, -12, -39}
;

const int16_t conv1d_135_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{56, -113, -38}
, {-137, -54, -58}
, {74, 49, -84}
, {-2, -68, -137}
, {-67, -114, 40}
, {60, -67, -114}
, {65, -33, 64}
, {20, -8, -161}
, {-80, -94, -29}
, {-105, -14, -28}
, {-47, 16, -6}
, {-14, -34, -81}
, {67, 52, 77}
, {47, -95, -6}
, {51, 37, -90}
, {17, 22, -29}
}
, {{6, -75, -104}
, {52, -90, -101}
, {33, -25, -40}
, {-15, -80, -137}
, {39, -37, 31}
, {-4, -105, 65}
, {63, -80, 17}
, {25, -103, -81}
, {-31, -34, -92}
, {5, 32, -94}
, {-50, -110, -18}
, {-15, -2, -37}
, {30, -5, -53}
, {-15, 46, -89}
, {-112, -10, 24}
, {-24, -90, -3}
}
, {{-117, -102, 88}
, {-9, -127, 37}
, {-61, -56, 61}
, {46, -79, -81}
, {-67, 13, 17}
, {-69, -132, 48}
, {93, 55, 75}
, {-10, 13, -43}
, {49, 4, 72}
, {-52, -45, -41}
, {-72, -16, -59}
, {-54, -30, -49}
, {-47, 32, -21}
, {-93, -11, 25}
, {-73, 4, -55}
, {-13, 30, 67}
}
, {{-145, -62, -68}
, {-25, 38, 72}
, {62, 104, 39}
, {-21, -89, 24}
, {-115, -97, -32}
, {-15, 3, -70}
, {-77, -21, 90}
, {80, -58, 56}
, {-9, 15, -10}
, {5, 28, 159}
, {-44, -79, -111}
, {-62, -87, 66}
, {-65, -57, 80}
, {-68, -73, 27}
, {5, 55, 82}
, {-73, -9, -2}
}
, {{-31, -9, -62}
, {17, 112, -21}
, {-31, -119, 8}
, {-109, -51, -112}
, {-110, -103, -81}
, {-33, -40, 34}
, {81, -56, -15}
, {78, -114, 10}
, {12, 3, -85}
, {26, -78, -80}
, {7, 121, -42}
, {-112, -50, -83}
, {28, 49, -86}
, {47, 17, -40}
, {76, -7, -70}
, {-4, -46, 21}
}
, {{-18, -116, -36}
, {0, -112, 37}
, {87, -102, 1}
, {-28, -43, -15}
, {57, -8, -89}
, {-13, 54, -127}
, {-58, -133, -94}
, {90, 74, 119}
, {46, -78, 75}
, {101, -78, 90}
, {25, -61, -30}
, {-22, 18, -38}
, {2, -121, -132}
, {-27, -7, -24}
, {-93, 49, 6}
, {-88, 47, -83}
}
, {{15, -112, -86}
, {-107, 11, -136}
, {28, -113, 32}
, {84, 8, -66}
, {44, -80, -91}
, {-140, -105, 13}
, {-129, -115, -12}
, {-123, -88, -149}
, {-133, 13, -61}
, {-125, 12, 23}
, {3, 90, -48}
, {-88, 20, -79}
, {33, -116, 49}
, {-132, 16, -10}
, {-139, -73, -34}
, {68, -29, 32}
}
, {{0, -123, 47}
, {27, 70, -1}
, {-50, -85, -115}
, {-70, 56, 63}
, {21, -34, -86}
, {-7, -16, 63}
, {-38, 2, 52}
, {-49, -28, -101}
, {-60, 50, -26}
, {45, -98, -71}
, {41, 95, 51}
, {-61, -59, 13}
, {-63, 30, -118}
, {6, -44, -80}
, {-133, -49, -69}
, {37, -110, -55}
}
, {{-7, -102, -104}
, {-14, -37, 52}
, {-98, -84, -94}
, {-18, 36, -73}
, {-88, 58, -45}
, {-8, -112, 26}
, {-17, -22, 1}
, {-91, -122, 60}
, {-40, -16, -32}
, {17, -36, -71}
, {43, 29, 2}
, {64, -36, 101}
, {-93, -146, -2}
, {-56, -14, -132}
, {-65, 0, -83}
, {-124, 46, -45}
}
, {{-73, -12, -59}
, {-51, -57, -84}
, {113, 54, -123}
, {-25, -97, -4}
, {-39, -26, -24}
, {-21, -7, -123}
, {-60, -28, 43}
, {48, -117, -52}
, {66, -14, -60}
, {38, -121, -9}
, {0, -90, -68}
, {-102, -98, 67}
, {114, 45, 78}
, {-107, -110, -8}
, {-41, 43, -23}
, {-56, 66, 46}
}
, {{-98, -33, -133}
, {48, -66, 23}
, {58, -8, -11}
, {-51, -168, -107}
, {21, 43, 69}
, {-32, 39, -13}
, {-52, -73, -51}
, {-18, 64, 59}
, {-96, -99, -78}
, {85, 51, -1}
, {-135, 12, 28}
, {71, -47, 25}
, {-83, -48, -124}
, {-119, 14, -117}
, {-94, -36, 48}
, {114, 21, -92}
}
, {{-85, -136, -82}
, {-110, 15, -138}
, {-28, -52, -34}
, {30, 14, -11}
, {-71, 19, -6}
, {-105, -98, -145}
, {-86, -73, -27}
, {-87, -33, 71}
, {-66, -54, 18}
, {56, 3, 94}
, {-58, -73, 28}
, {-124, 50, 17}
, {58, -60, 34}
, {-53, 0, -80}
, {5, -45, 44}
, {-35, 36, 19}
}
, {{-73, -105, -11}
, {-21, -87, -32}
, {16, -82, 17}
, {-60, -37, 4}
, {13, -20, 40}
, {68, -50, -64}
, {29, -26, 61}
, {-18, 43, 5}
, {-80, -8, -106}
, {-8, 110, 34}
, {24, 50, -52}
, {53, 51, 68}
, {30, -30, -97}
, {-123, -18, -6}
, {-89, -91, -65}
, {57, 128, 87}
}
, {{-12, -35, -71}
, {65, -20, 61}
, {38, -108, -55}
, {-60, -91, 25}
, {84, 66, 73}
, {53, -6, -46}
, {65, -70, -61}
, {119, 27, 61}
, {36, -66, 86}
, {-106, 42, -58}
, {5, -56, 22}
, {77, -42, 0}
, {-52, -5, -49}
, {-119, -64, -3}
, {5, 58, 32}
, {78, 48, 58}
}
, {{-36, -123, 92}
, {-111, 19, 190}
, {-85, 46, -64}
, {-84, -131, -120}
, {-155, -38, 137}
, {25, 93, 96}
, {-32, -68, 69}
, {-10, 43, 91}
, {-22, 25, -36}
, {35, 62, 60}
, {-9, -108, 28}
, {-74, 32, 82}
, {-26, -115, 29}
, {-110, 67, -76}
, {-107, 49, -80}
, {39, 96, -78}
}
, {{-1, -26, -14}
, {-79, 71, -50}
, {-117, -118, -110}
, {-117, -58, -83}
, {-47, 3, 43}
, {90, -101, 17}
, {-1, -71, -61}
, {-50, -128, -136}
, {-132, 41, -15}
, {-27, -125, -111}
, {87, 68, -64}
, {58, 19, -113}
, {-34, 13, -3}
, {-24, -105, -23}
, {-13, 46, -20}
, {48, 103, -62}
}
, {{-48, -142, 27}
, {-53, 16, -93}
, {-4, 47, -85}
, {27, -90, 53}
, {0, 8, -39}
, {-18, -122, 24}
, {-14, -10, -101}
, {-82, 30, 92}
, {7, -125, 1}
, {-91, -71, 48}
, {6, 28, 84}
, {41, -13, -29}
, {-124, -49, 0}
, {-121, -110, 107}
, {-101, -3, 3}
, {21, 35, 44}
}
, {{89, -25, -155}
, {2, -98, -51}
, {50, 63, -125}
, {-82, -9, -82}
, {-1, -93, -6}
, {-118, -161, -85}
, {12, 39, 25}
, {34, -73, -61}
, {27, -47, -9}
, {1, -5, -1}
, {-99, -24, -75}
, {-16, -9, 113}
, {-107, -21, -40}
, {-151, 11, 45}
, {-110, -50, -102}
, {-107, -3, -101}
}
, {{-51, -127, -84}
, {-98, -8, 0}
, {83, 100, -74}
, {52, 41, -142}
, {-74, 12, -65}
, {67, -81, 28}
, {-98, 33, 12}
, {51, -43, -73}
, {80, -16, 48}
, {21, 20, 10}
, {-23, -25, -130}
, {20, -16, -60}
, {-73, 51, -81}
, {-22, -106, -77}
, {-71, 10, -44}
, {-24, -11, -87}
}
, {{-38, -17, -6}
, {-95, -49, -16}
, {-86, 63, -51}
, {12, 15, -81}
, {-134, 81, -113}
, {-7, 52, 9}
, {31, -13, 25}
, {-81, -97, 54}
, {-82, -20, 21}
, {-104, 37, -14}
, {46, -110, 0}
, {-4, -85, -25}
, {0, -111, -61}
, {-49, -13, -101}
, {20, -13, 19}
, {6, -107, -60}
}
, {{6, -16, -104}
, {7, -42, 51}
, {-125, -78, 45}
, {18, 46, -129}
, {-103, -46, -26}
, {45, -57, 85}
, {60, 70, 51}
, {-48, 78, 11}
, {63, -26, 44}
, {-11, 129, 54}
, {-111, 29, -71}
, {-106, -33, -48}
, {-105, -103, -88}
, {23, -68, -130}
, {27, 33, -106}
, {-34, -99, 82}
}
, {{-10, 95, -77}
, {-101, -55, 111}
, {67, 69, -67}
, {13, -24, -43}
, {-77, -32, 69}
, {-65, -111, 13}
, {-2, 20, -78}
, {14, -65, -39}
, {-80, -8, 97}
, {-20, -38, -68}
, {-64, 24, -80}
, {42, 36, 8}
, {-61, -81, -83}
, {-124, -15, -97}
, {79, 53, -110}
, {28, -62, 65}
}
, {{4, -108, -53}
, {-91, -115, -26}
, {-89, 10, 104}
, {-47, 22, -44}
, {52, -81, 93}
, {-33, 14, 10}
, {-35, -60, 74}
, {47, -10, 15}
, {26, -67, 0}
, {85, 0, 17}
, {48, -106, 62}
, {-43, 56, -46}
, {-111, -28, 11}
, {-22, -99, 47}
, {-43, 16, -67}
, {112, 8, 55}
}
, {{24, -98, 19}
, {-19, -115, -17}
, {-50, -63, -25}
, {-34, -125, -13}
, {34, 36, 65}
, {-102, -117, -18}
, {60, -86, 21}
, {-45, 48, 50}
, {-33, -130, -66}
, {-80, 38, -11}
, {-54, -24, -72}
, {70, 38, -107}
, {-40, -95, -29}
, {-111, -99, -8}
, {-46, -124, 10}
, {0, -102, -34}
}
, {{-40, 29, 103}
, {-53, -92, -107}
, {-19, -12, -78}
, {-92, 31, -90}
, {-10, -94, -142}
, {43, 15, -61}
, {-99, -57, -96}
, {-96, -90, 45}
, {39, 36, -130}
, {-76, -58, 48}
, {75, -30, 63}
, {-101, 34, 31}
, {65, 12, -107}
, {-106, 19, -115}
, {-123, -58, 50}
, {51, -74, -100}
}
, {{-84, -24, -127}
, {6, -109, -36}
, {-25, -15, -124}
, {-44, 23, 41}
, {-38, -119, -60}
, {-69, -3, 39}
, {-5, 50, 15}
, {-64, 97, -25}
, {28, -97, -81}
, {69, -51, -79}
, {-15, -125, 68}
, {-115, -53, -5}
, {-27, 17, -81}
, {-77, -46, 35}
, {-133, -93, -81}
, {2, -54, -34}
}
, {{11, 78, -19}
, {-26, 8, 27}
, {-109, 58, 38}
, {-29, -146, -107}
, {-131, -115, -33}
, {47, -51, -22}
, {-95, 67, -49}
, {3, -117, -130}
, {-18, -81, -32}
, {14, 42, 15}
, {-1, -39, -149}
, {-43, 51, -95}
, {-124, 17, -117}
, {41, -148, -94}
, {-35, -76, -91}
, {-114, -13, -59}
}
, {{8, -131, 20}
, {32, 15, -79}
, {-43, -93, -99}
, {-93, -154, -81}
, {13, -46, -86}
, {-137, -67, 18}
, {-54, 25, -92}
, {85, -19, -81}
, {102, 5, -33}
, {64, 9, 94}
, {36, -76, -17}
, {-23, 21, 67}
, {-18, -46, 35}
, {-39, -130, -71}
, {-58, -128, 5}
, {0, -19, 108}
}
, {{71, 46, -97}
, {63, -68, -65}
, {9, -125, 38}
, {40, 112, -76}
, {-1, -74, -52}
, {-11, -21, 63}
, {63, 55, -90}
, {-5, 84, 67}
, {-99, -32, -102}
, {17, -107, 1}
, {-22, 70, 66}
, {-78, 33, -25}
, {-33, -35, -44}
, {15, 44, -4}
, {-39, 33, -79}
, {63, 55, -127}
}
, {{92, -34, 114}
, {-30, -178, -159}
, {-66, -76, 103}
, {-11, -35, 59}
, {-70, 67, 83}
, {-97, 64, 10}
, {12, -97, -83}
, {-121, -24, -22}
, {58, 23, -11}
, {-48, -101, -9}
, {55, -5, 78}
, {-81, 10, 110}
, {36, 5, -9}
, {-65, 100, -65}
, {78, -55, 75}
, {-42, 37, 31}
}
, {{-52, -86, -132}
, {-46, -130, 0}
, {78, -24, 92}
, {118, -47, -9}
, {-14, -88, -116}
, {72, -84, 77}
, {-116, 53, -63}
, {76, -101, 54}
, {1, -82, -107}
, {-71, -1, -72}
, {75, -12, 113}
, {63, -52, -69}
, {41, -48, -69}
, {-106, -11, 28}
, {-12, 23, 0}
, {-120, -3, 22}
}
, {{-83, -12, -108}
, {20, -73, -23}
, {27, 96, -107}
, {62, -118, 12}
, {-78, 28, -14}
, {-79, -119, -65}
, {34, -19, 97}
, {53, 85, -67}
, {-59, -108, 18}
, {-42, -90, -42}
, {-40, -139, 24}
, {-6, 104, -85}
, {-23, -22, 60}
, {-6, -56, 41}
, {-42, -3, -63}
, {8, -118, -111}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE