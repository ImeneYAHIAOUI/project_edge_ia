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


const int16_t conv1d_98_bias[CONV_FILTERS] = {15, -8, -68, -47, -11, 6, -64, -46, -52, 144, -36, -58, 73, -45, -39, -47, 74, -27, -7, -48, -28, -31, 2, -61, 60, -19, 4, -55, -48, 81, 51, 7}
;

const int16_t conv1d_98_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-69, -6, -121}
, {-85, -80, -74}
, {-135, 27, -112}
, {67, 149, -132}
, {-104, 28, 3}
, {-60, 38, 27}
, {-27, 7, 50}
, {27, -141, -52}
, {1, -48, -102}
, {-37, -11, -28}
, {46, -56, 95}
, {-29, -19, 1}
, {-96, -67, -54}
, {13, 43, -6}
, {90, -35, 101}
, {-85, -33, 83}
}
, {{-46, 104, 2}
, {-17, -3, -38}
, {-93, -14, -106}
, {65, 100, -70}
, {4, 69, -64}
, {-2, -31, -104}
, {76, -104, 19}
, {-83, 29, 73}
, {23, -33, -70}
, {-130, -79, 1}
, {80, -60, 58}
, {-58, 129, -100}
, {-155, -50, -56}
, {66, -73, -127}
, {-13, -53, -133}
, {43, 48, 108}
}
, {{-109, -82, 18}
, {21, -54, 57}
, {-40, 13, -47}
, {-4, -29, -146}
, {68, -96, 63}
, {-3, 56, -88}
, {0, 29, -58}
, {7, 27, 20}
, {23, -107, 54}
, {98, -22, 37}
, {2, -42, -102}
, {28, -27, -48}
, {21, -96, 7}
, {62, -41, 20}
, {27, 86, -6}
, {-103, -20, -2}
}
, {{-22, 41, 18}
, {-29, 27, 0}
, {19, -61, -80}
, {0, -128, -62}
, {-94, -45, -70}
, {30, -138, -150}
, {-129, -25, -48}
, {-102, 26, 76}
, {23, 13, -79}
, {-153, 39, -126}
, {19, 50, 51}
, {52, -93, -117}
, {48, 33, 19}
, {41, -139, 69}
, {17, -149, 55}
, {-67, -29, -18}
}
, {{-100, -12, 29}
, {-98, -95, -67}
, {27, 66, -123}
, {76, -96, -62}
, {-7, 8, 35}
, {19, -78, -113}
, {10, -49, -76}
, {30, -90, -80}
, {-103, -115, -71}
, {-90, -110, 32}
, {-134, 66, -82}
, {-91, -56, 50}
, {72, -28, -87}
, {-16, 6, 61}
, {-69, -136, 3}
, {-117, 13, -129}
}
, {{-115, 60, -25}
, {73, 112, -15}
, {32, -41, 7}
, {112, 143, 33}
, {-127, 50, 39}
, {-23, -30, 28}
, {23, 41, -98}
, {-29, -114, -44}
, {-128, 35, -140}
, {69, -11, -114}
, {17, 59, -76}
, {-89, -28, 7}
, {-14, -95, -139}
, {65, 59, 37}
, {-32, -15, -206}
, {-86, -73, -149}
}
, {{63, -95, -88}
, {-101, -133, -76}
, {-96, 11, 19}
, {139, -24, -17}
, {91, -52, 11}
, {-60, 17, 35}
, {23, 86, 30}
, {-59, -92, 18}
, {40, -16, 9}
, {44, 89, -11}
, {-136, 52, 37}
, {-42, -75, 87}
, {-99, 32, 56}
, {-107, 22, 57}
, {29, -114, -1}
, {-14, 21, -78}
}
, {{42, 16, 33}
, {79, -27, -75}
, {26, 15, 22}
, {72, 7, -54}
, {-61, -45, 15}
, {-130, -41, -63}
, {-44, 40, -126}
, {50, -32, -95}
, {-116, 0, -20}
, {-122, -75, -147}
, {0, 64, -114}
, {34, -109, 3}
, {61, -126, -39}
, {-104, -14, -140}
, {-144, 54, -142}
, {-134, -122, -108}
}
, {{-18, 11, 69}
, {77, -64, -20}
, {-26, -79, 29}
, {42, 42, 36}
, {43, -97, -96}
, {-24, 9, 3}
, {92, -74, -81}
, {123, -33, -32}
, {15, -81, 38}
, {-81, 43, -71}
, {-129, -6, -140}
, {-6, -77, 65}
, {-98, -117, 48}
, {73, 68, -31}
, {-28, -49, 39}
, {15, 53, -12}
}
, {{90, 77, -86}
, {66, -108, 132}
, {-57, -56, -39}
, {6, 126, 38}
, {1, -121, 77}
, {77, -6, 106}
, {-6, 60, -49}
, {-1, 117, 55}
, {-89, -136, 28}
, {-20, -29, 0}
, {-44, 74, -24}
, {-166, 42, -64}
, {46, 176, -109}
, {-24, -15, 48}
, {-80, 95, 32}
, {-39, 34, -60}
}
, {{-29, 71, -47}
, {-79, -92, -32}
, {-145, -96, -47}
, {60, -135, -94}
, {-114, -77, -139}
, {7, -88, 9}
, {61, -47, 87}
, {-11, -16, -52}
, {-31, -133, 33}
, {-81, -119, -123}
, {34, -18, -114}
, {-53, -90, 74}
, {1, -98, 9}
, {-72, 35, -66}
, {-150, -47, -32}
, {-52, -37, 11}
}
, {{53, -48, 12}
, {-10, -41, -146}
, {-111, -149, 21}
, {107, -87, 27}
, {47, -113, -124}
, {-2, -134, -34}
, {-101, -95, -119}
, {16, -34, 11}
, {-167, -85, -98}
, {-70, -185, -148}
, {84, -84, -18}
, {-141, 119, -19}
, {-84, -80, 45}
, {43, -74, -141}
, {-83, -50, -7}
, {-156, -8, 1}
}
, {{47, -99, -71}
, {-97, -10, 70}
, {0, -65, 37}
, {64, 96, 119}
, {29, -62, 61}
, {51, -71, -9}
, {-61, -67, -57}
, {-71, -100, 40}
, {-125, -36, -31}
, {-69, -37, -4}
, {75, 54, 61}
, {19, -60, 60}
, {-10, -21, 48}
, {82, 55, 63}
, {43, -3, -22}
, {-11, -55, -112}
}
, {{-12, 23, -66}
, {-9, 34, -105}
, {63, 38, 29}
, {-151, -19, -6}
, {-48, -17, 96}
, {88, 0, 21}
, {-6, -63, 109}
, {-30, -64, -36}
, {-20, 104, 64}
, {27, 28, -58}
, {-51, -35, 60}
, {59, 10, -9}
, {69, -17, 124}
, {-33, -19, -52}
, {3, 82, -44}
, {-82, 61, 31}
}
, {{-77, -96, -103}
, {-118, 4, 31}
, {11, -19, -64}
, {-83, 44, 114}
, {-15, -65, 60}
, {-76, -44, 63}
, {-2, 15, -20}
, {69, -41, 67}
, {-20, 55, -1}
, {-18, -14, -26}
, {4, -123, -82}
, {-49, -47, -88}
, {-29, -110, -94}
, {-101, 20, -15}
, {-11, -125, -13}
, {-129, -24, -91}
}
, {{-131, -90, 0}
, {-175, -63, 23}
, {-48, -29, -86}
, {-89, -12, 0}
, {-5, -23, 6}
, {28, 59, -27}
, {49, -74, -71}
, {15, -73, 14}
, {-64, -1, 18}
, {-138, -127, -133}
, {-69, -117, -2}
, {93, -125, -43}
, {30, -104, -3}
, {-92, -16, 9}
, {6, -103, -38}
, {-41, -1, -110}
}
, {{-27, 64, 43}
, {-116, -139, -91}
, {-120, 6, 48}
, {94, 139, 127}
, {-29, -22, 36}
, {9, 2, -102}
, {24, 56, -103}
, {46, -132, 68}
, {40, -40, -120}
, {-93, -108, -127}
, {-62, 67, 34}
, {-78, 2, -100}
, {18, -67, -67}
, {7, 34, -76}
, {-101, -22, 64}
, {-15, -27, -38}
}
, {{18, -44, -25}
, {120, 3, -81}
, {21, -11, 33}
, {-39, -87, -81}
, {-62, -122, -78}
, {57, -27, 53}
, {-51, -12, 86}
, {69, 31, 56}
, {25, -143, -76}
, {-41, -96, -80}
, {-77, -47, -47}
, {7, -105, -60}
, {23, -51, 17}
, {-111, -59, -54}
, {-67, -45, -47}
, {-28, -63, 3}
}
, {{-48, 62, -77}
, {-67, -69, -48}
, {-108, -120, -123}
, {87, 46, -109}
, {67, -26, -73}
, {-49, 56, -71}
, {98, -46, 95}
, {8, -61, 102}
, {-41, -106, -147}
, {-127, 6, 94}
, {-15, -14, -44}
, {-87, 46, -128}
, {-4, 74, 72}
, {45, 75, -104}
, {-72, -118, 22}
, {-123, 52, -28}
}
, {{-70, -43, -69}
, {37, 23, -56}
, {51, -78, -92}
, {-85, -90, 26}
, {34, -136, -35}
, {-27, -40, 65}
, {61, 101, 49}
, {-35, 51, -60}
, {88, -138, -91}
, {52, 101, 71}
, {-87, 12, -53}
, {46, -136, 4}
, {-93, 16, 51}
, {6, -110, -85}
, {91, -51, -8}
, {-75, 5, 46}
}
, {{29, 6, 19}
, {6, 65, -119}
, {-99, -10, -35}
, {-6, 54, -36}
, {-138, -10, -34}
, {-123, 13, -28}
, {-46, -105, 8}
, {-82, -40, -4}
, {-119, -92, 13}
, {-98, -32, 0}
, {44, 35, -105}
, {-1, 43, -54}
, {-102, -35, -115}
, {20, -6, -67}
, {-55, -59, 6}
, {-49, -46, -23}
}
, {{-117, 55, 30}
, {-81, 56, -99}
, {8, 37, -123}
, {6, 34, -108}
, {-55, -18, -13}
, {91, -28, 46}
, {-115, -11, 29}
, {-34, -80, -45}
, {21, -97, -15}
, {-77, 80, -25}
, {-76, 0, 62}
, {-106, 64, -19}
, {-57, -77, 60}
, {-115, -36, -63}
, {-38, -11, 45}
, {74, -25, -74}
}
, {{8, 26, -73}
, {-206, -34, 27}
, {-125, -12, -100}
, {226, -38, -157}
, {-89, -121, -110}
, {-65, 44, -29}
, {63, -61, -106}
, {-42, 8, 13}
, {-112, -56, -135}
, {-77, 72, 82}
, {34, -108, -46}
, {-91, -104, -42}
, {55, 7, -85}
, {60, -54, 70}
, {-89, 21, -79}
, {-126, 17, 111}
}
, {{-20, -89, 84}
, {60, 19, -133}
, {-33, -63, 53}
, {93, 25, -51}
, {92, -62, -8}
, {-56, 6, 28}
, {-93, 0, 49}
, {55, -108, -44}
, {112, -28, -54}
, {-59, 63, -47}
, {74, -12, -53}
, {103, -88, -21}
, {-89, 38, 37}
, {59, 1, 19}
, {-67, 42, -1}
, {97, 82, 50}
}
, {{92, -38, -102}
, {-62, 89, 23}
, {2, 55, -80}
, {-5, -54, 82}
, {-120, 47, -81}
, {-88, -85, -130}
, {0, -2, 122}
, {53, -23, 28}
, {63, -27, -62}
, {5, -6, -111}
, {-76, -78, 48}
, {17, 51, -21}
, {-110, 1, -87}
, {-67, 15, 49}
, {17, 43, -163}
, {-24, 41, -131}
}
, {{-117, 52, 18}
, {179, -99, 47}
, {-123, -113, -45}
, {-36, -46, -70}
, {-103, -84, 50}
, {62, -84, -16}
, {64, 29, -91}
, {-88, 118, 50}
, {-28, -53, 13}
, {23, 87, 117}
, {48, 30, 50}
, {-11, -114, -107}
, {60, 63, 44}
, {-21, 55, -60}
, {118, -88, -55}
, {-56, 102, -87}
}
, {{-64, -64, -58}
, {40, -62, 0}
, {-86, -70, -48}
, {-199, -39, -94}
, {-52, 5, -118}
, {85, -173, -73}
, {42, -63, 44}
, {-32, 9, 7}
, {74, -5, 48}
, {-146, -63, -127}
, {103, -66, -9}
, {-49, 46, -154}
, {-66, 78, 48}
, {-48, -71, -120}
, {-15, 4, 0}
, {70, 41, 57}
}
, {{79, -84, -42}
, {126, -119, 87}
, {-12, 25, 117}
, {-181, -179, -113}
, {-79, -36, -56}
, {140, -37, -75}
, {3, 81, 131}
, {-1, -5, -19}
, {96, -39, 77}
, {-23, 43, -88}
, {-131, 24, -19}
, {-10, -35, 35}
, {35, 15, -64}
, {-43, -12, 46}
, {-6, -8, 38}
, {-47, 12, -15}
}
, {{-123, -111, -4}
, {-6, -28, -88}
, {41, -42, 59}
, {-95, -128, -77}
, {-59, -42, 48}
, {-32, 35, -142}
, {45, -118, -18}
, {104, 66, -2}
, {-111, 47, 5}
, {-33, 62, 9}
, {-135, -18, 33}
, {78, -85, -96}
, {29, -104, 48}
, {-28, -85, 22}
, {-7, -10, -67}
, {-79, -104, -121}
}
, {{38, 43, 64}
, {-25, 22, 0}
, {8, 69, 62}
, {142, 0, 50}
, {33, 70, -127}
, {75, 47, -46}
, {114, -83, 71}
, {-61, -21, -67}
, {-75, -109, 41}
, {-39, -53, -49}
, {-121, 0, -144}
, {23, 70, -92}
, {16, 48, -133}
, {-55, -106, -66}
, {-81, -115, -11}
, {-92, 33, -6}
}
, {{-118, 65, -15}
, {-49, 106, 97}
, {-8, -98, 39}
, {112, 159, -7}
, {-97, -97, -100}
, {81, -56, 54}
, {74, 5, -103}
, {-51, 73, 87}
, {-83, -75, -115}
, {43, 7, -90}
, {-65, 81, -50}
, {-50, 34, -78}
, {-62, -13, -105}
, {-111, 81, -42}
, {37, -48, -86}
, {-66, -30, -182}
}
, {{73, -71, 10}
, {-86, -111, -79}
, {6, -74, 91}
, {1, 98, -37}
, {-84, -49, 16}
, {64, 14, 19}
, {-81, -96, -76}
, {-60, -117, 34}
, {19, -45, -58}
, {-16, -33, 47}
, {-71, -44, -28}
, {69, 10, 46}
, {-2, -82, 26}
, {-111, -13, 52}
, {84, 108, -3}
, {-61, 50, 103}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE