/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    20
#define CONV_FILTERS      30
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_65_bias[CONV_FILTERS] = {-13, -32, -25, -11, -29, -14, 21, -7, -30, -42, -3, 25, -10, 27, 15, -19, -10, -35, 23, 5, -13, 38, 21, 18, 2, -24, 36, 5, -9, 28}
;

const int16_t conv1d_65_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-35, -38, 70}
, {111, -47, -44}
, {-77, -76, -89}
, {-67, 127, -32}
, {-1, 38, 62}
, {-16, -125, 21}
, {-54, -34, 101}
, {-9, -40, 93}
, {83, 38, 38}
, {-12, 139, -3}
, {68, -27, -19}
, {-6, 63, 46}
, {10, 98, -26}
, {13, 93, 29}
, {-25, -105, 21}
, {96, 71, -41}
, {-91, -50, -36}
, {-54, 28, 42}
, {-20, 83, 79}
, {-19, 122, -28}
}
, {{-52, -13, 61}
, {68, 88, -2}
, {27, 12, 20}
, {-69, 81, 54}
, {26, -143, -69}
, {-1, -101, -24}
, {31, -61, 3}
, {124, -39, 77}
, {-85, 30, 52}
, {-16, -52, 79}
, {-44, -72, 111}
, {-35, 55, -73}
, {61, 51, -53}
, {46, -12, 4}
, {52, 59, 81}
, {74, -1, 54}
, {85, 43, -89}
, {-120, 0, 72}
, {54, 89, 45}
, {-2, 95, 0}
}
, {{-56, 81, -19}
, {-27, 88, -1}
, {-22, 93, 62}
, {161, 7, 79}
, {-24, 58, -65}
, {-31, 17, 13}
, {54, 12, 56}
, {-81, 18, -117}
, {-97, 87, 27}
, {16, -100, -54}
, {5, -95, -112}
, {65, 49, -11}
, {-131, 90, 58}
, {-63, 28, 127}
, {14, 34, 111}
, {-21, 38, -99}
, {43, 46, -78}
, {-37, -94, -87}
, {-55, 54, -63}
, {-66, 34, -18}
}
, {{43, -86, -14}
, {9, 1, -7}
, {-47, -43, -108}
, {25, 137, 3}
, {57, -102, 69}
, {50, -46, -33}
, {76, 37, 50}
, {33, -50, -116}
, {11, 5, 70}
, {87, -82, -106}
, {36, 108, -20}
, {58, 4, 62}
, {38, 5, 10}
, {46, 76, -101}
, {-76, -6, 39}
, {-1, 81, -88}
, {-19, 66, -66}
, {-52, -47, 11}
, {-31, 17, 85}
, {105, 19, -19}
}
, {{-66, 88, -58}
, {-100, -25, -77}
, {-78, 12, 1}
, {-10, -1, -65}
, {-29, -44, 58}
, {42, -54, -22}
, {99, -96, 107}
, {-65, 12, 57}
, {92, 6, -45}
, {30, -25, 114}
, {-39, 2, 41}
, {-38, -65, 6}
, {4, -30, -26}
, {-83, 8, 29}
, {-87, 31, 0}
, {-41, -64, -104}
, {-78, -33, 29}
, {-81, -61, 104}
, {102, 87, -39}
, {-12, -25, -31}
}
, {{-5, -55, -61}
, {88, 65, -113}
, {-13, -80, 99}
, {-100, -54, 2}
, {78, 2, 52}
, {-84, 70, -100}
, {-89, 34, -79}
, {-4, 120, 143}
, {-80, -92, 62}
, {81, 109, 27}
, {-15, 68, -29}
, {-27, 29, -91}
, {-3, -58, -56}
, {-26, -20, 25}
, {-72, -40, -53}
, {88, 3, -19}
, {12, 111, -67}
, {6, 94, 55}
, {65, 63, 94}
, {-67, 100, -39}
}
, {{-15, -49, 49}
, {106, 42, 0}
, {13, 59, 0}
, {22, 24, -70}
, {-27, 77, 150}
, {40, -19, 91}
, {84, 70, 74}
, {-73, -10, 0}
, {119, 82, -89}
, {-29, -5, 45}
, {-86, -106, 77}
, {-83, 28, 62}
, {64, -30, -78}
, {105, 100, -74}
, {77, -49, 23}
, {48, -100, 89}
, {73, -23, -107}
, {97, -79, 99}
, {-22, -41, 20}
, {-58, 84, -33}
}
, {{-97, 29, -81}
, {-56, -67, -4}
, {31, 22, 79}
, {35, -90, 76}
, {-10, -87, 142}
, {-25, -75, 78}
, {-71, -49, 95}
, {11, 104, -30}
, {-73, 71, 46}
, {25, -6, 9}
, {-17, -33, 13}
, {-56, 13, -55}
, {-48, 90, -51}
, {13, 0, 16}
, {36, 22, 14}
, {85, 10, 43}
, {-43, -46, -59}
, {42, 0, -96}
, {37, -8, -17}
, {32, 52, -40}
}
, {{-96, 49, -75}
, {-41, -66, -50}
, {86, 91, -93}
, {66, 59, -126}
, {-85, 13, 50}
, {37, -35, -71}
, {49, -16, 102}
, {79, 63, -80}
, {83, 28, 4}
, {62, 91, -38}
, {-18, 67, 81}
, {-84, 0, 17}
, {-89, 35, 33}
, {90, -17, -81}
, {-26, -83, 1}
, {91, -32, -92}
, {36, 26, 93}
, {94, -81, -22}
, {-11, 35, -87}
, {6, 102, 72}
}
, {{28, -50, -99}
, {-59, -56, -81}
, {26, 90, 18}
, {-95, 49, -33}
, {26, 67, -39}
, {87, 19, -28}
, {-104, -30, 19}
, {-37, 82, 31}
, {13, -52, -48}
, {1, -84, -14}
, {52, 85, -16}
, {-2, -6, -95}
, {-37, 19, -63}
, {8, 50, 16}
, {23, 8, -93}
, {16, 102, 79}
, {-94, -36, -104}
, {-97, -71, -76}
, {-13, -90, -47}
, {69, 76, 9}
}
, {{82, 56, -36}
, {-27, 78, 25}
, {-80, -14, -30}
, {-98, -88, -36}
, {-20, -72, -52}
, {39, 28, -48}
, {-38, 4, -21}
, {-52, -51, -103}
, {-49, -37, 96}
, {-100, -20, -19}
, {86, -84, -75}
, {-6, -39, -56}
, {-36, 13, 23}
, {-51, 48, 88}
, {-45, -50, 16}
, {-98, -66, -58}
, {-78, 81, 87}
, {86, -45, -86}
, {-62, -93, -55}
, {-74, 92, 30}
}
, {{105, 93, 4}
, {19, 47, -62}
, {12, 66, -52}
, {-59, -33, 52}
, {96, -119, -15}
, {105, 67, 117}
, {-75, 20, 12}
, {78, -75, 0}
, {44, 2, -100}
, {-28, -99, -80}
, {-56, 17, -84}
, {-37, 43, 86}
, {19, 91, 7}
, {-14, -40, -28}
, {0, -29, 88}
, {-97, 59, 87}
, {-78, 49, 90}
, {95, 103, 24}
, {-56, 15, -42}
, {-22, -94, -49}
}
, {{-61, -65, 18}
, {44, 66, -35}
, {-87, 9, -73}
, {43, -35, -34}
, {-76, -80, -63}
, {54, -23, 94}
, {-27, 28, -26}
, {64, 84, 36}
, {-95, 83, -4}
, {-36, 36, -13}
, {39, -48, -81}
, {62, -38, 81}
, {2, -41, -91}
, {30, 10, 71}
, {-17, -119, 71}
, {-46, -44, -97}
, {59, 50, -100}
, {-6, -72, -90}
, {-81, -86, 96}
, {77, 79, 70}
}
, {{148, -39, 54}
, {59, -19, 42}
, {68, 109, 133}
, {-55, -6, -44}
, {78, 50, 72}
, {-46, -85, 11}
, {88, 65, -51}
, {-104, 20, -22}
, {-27, -53, 121}
, {0, -67, -14}
, {43, -56, -36}
, {-65, -12, 16}
, {-12, 68, 8}
, {91, -92, 27}
, {-106, 40, -62}
, {32, 8, -78}
, {-20, -53, 87}
, {-39, 49, 19}
, {45, 15, 0}
, {18, 53, 24}
}
, {{107, 91, 81}
, {-87, -24, -50}
, {-46, 29, 120}
, {-11, -115, -14}
, {57, 53, -26}
, {7, -31, -71}
, {0, -20, 88}
, {-36, 9, 61}
, {-98, 9, -43}
, {-26, -51, 99}
, {-46, 12, -62}
, {13, 60, -95}
, {79, -3, 71}
, {-51, -48, 58}
, {41, 35, -133}
, {-69, 55, 5}
, {75, -48, 70}
, {-33, 84, 17}
, {45, 73, 38}
, {50, -8, -89}
}
, {{-6, 19, -52}
, {80, -9, 67}
, {57, -84, -29}
, {9, -7, 86}
, {-111, 14, -101}
, {-98, -110, -157}
, {85, -26, 23}
, {-52, -88, 75}
, {50, -2, -94}
, {74, 82, 4}
, {-25, 112, 91}
, {29, -42, -10}
, {-68, 62, 0}
, {-104, 98, 6}
, {83, 103, -3}
, {13, 94, 76}
, {-16, 26, 74}
, {-80, 71, 49}
, {48, 118, 67}
, {0, 0, 103}
}
, {{76, -2, 47}
, {-96, 42, -51}
, {-51, -10, -50}
, {-62, 40, 11}
, {-83, 48, 84}
, {-63, 57, 1}
, {-92, -44, 19}
, {1, -21, 61}
, {38, -54, 19}
, {36, 17, 6}
, {-77, 78, 50}
, {-88, -80, -56}
, {33, -66, -53}
, {-63, -52, -42}
, {-51, 60, -22}
, {92, -44, 45}
, {-32, 4, -108}
, {-16, -5, -2}
, {-110, 15, 5}
, {-25, -32, -104}
}
, {{36, -105, 73}
, {10, -10, 3}
, {9, -1, 56}
, {-80, -86, -50}
, {77, 63, 37}
, {106, 5, 130}
, {-4, 94, 54}
, {19, 140, -39}
, {26, -117, -14}
, {-50, 130, 100}
, {-7, 19, -59}
, {-77, 53, 11}
, {85, -4, 84}
, {-1, 99, -13}
, {-83, -107, -48}
, {44, -34, -58}
, {103, -21, 46}
, {-8, -39, 64}
, {16, 3, -71}
, {10, -16, -32}
}
, {{101, -65, 140}
, {69, -37, -33}
, {-22, 103, 122}
, {-80, 56, -21}
, {151, 107, -13}
, {6, 161, 78}
, {103, 56, 93}
, {3, 33, 91}
, {105, -65, 8}
, {43, -63, -67}
, {-59, 114, -4}
, {90, -37, 32}
, {60, -90, -18}
, {63, -125, -35}
, {-71, -70, -34}
, {8, 104, -75}
, {-31, -71, -89}
, {59, 23, -46}
, {10, -40, -79}
, {-99, -54, 3}
}
, {{-60, 6, 9}
, {-50, 26, 39}
, {-1, -43, 122}
, {-84, 20, -50}
, {133, -17, -42}
, {-4, 95, 112}
, {33, -1, -63}
, {46, 33, 106}
, {-57, 31, 86}
, {-58, 43, -14}
, {61, -22, 63}
, {-55, -25, -2}
, {7, 79, -9}
, {-98, -18, 2}
, {16, -93, -31}
, {42, -107, 83}
, {-64, 32, -85}
, {-69, 102, 29}
, {-119, 53, -6}
, {-19, -101, -4}
}
, {{-72, -40, 33}
, {26, 39, 65}
, {60, 13, -34}
, {-2, 93, 63}
, {55, 50, -15}
, {66, -40, -169}
, {-60, -50, -27}
, {47, -15, -9}
, {91, -79, 10}
, {48, 27, 53}
, {92, 87, 65}
, {93, -67, 69}
, {86, -63, 60}
, {-67, 86, -88}
, {54, -112, 35}
, {13, -2, 49}
, {19, 69, 20}
, {-95, 43, 7}
, {92, 106, -29}
, {-32, -36, 87}
}
, {{6, 87, -76}
, {77, 102, 22}
, {5, 103, -18}
, {67, -104, 29}
, {-4, -24, -58}
, {-113, 60, -53}
, {61, -50, -43}
, {22, 0, 37}
, {112, 107, 9}
, {-68, 76, -82}
, {-79, 41, 58}
, {88, 90, -28}
, {11, 39, 77}
, {7, -39, 41}
, {-31, -64, 89}
, {94, -66, -9}
, {62, 61, -80}
, {111, -48, 40}
, {-46, -72, -79}
, {-19, 0, 18}
}
, {{29, -44, 21}
, {9, -51, -39}
, {-63, -112, 63}
, {52, -39, 19}
, {-125, -143, -101}
, {32, -65, -59}
, {-59, 59, -100}
, {-1, -46, -15}
, {-85, -69, -22}
, {71, -77, -61}
, {28, 76, 112}
, {18, -87, 3}
, {118, -86, 3}
, {-8, -86, -20}
, {87, 40, 95}
, {22, -18, 98}
, {-16, 82, -3}
, {-71, 82, -62}
, {7, -44, 100}
, {-23, 104, 10}
}
, {{62, -45, -5}
, {20, 53, -55}
, {-125, -89, -41}
, {73, -10, 97}
, {-102, -3, -149}
, {-38, 7, -92}
, {72, -56, 52}
, {-91, 70, -5}
, {64, -17, -43}
, {8, 48, 5}
, {55, -14, -10}
, {90, 9, -65}
, {30, 81, -49}
, {-100, 61, -102}
, {12, -55, 16}
, {88, -61, -92}
, {96, 14, 105}
, {50, -75, -30}
, {78, 9, -63}
, {-20, 0, 36}
}
, {{-22, -79, -89}
, {-92, -112, -102}
, {61, -60, -103}
, {-23, 91, 0}
, {-71, -22, -109}
, {-142, -55, 22}
, {29, -84, 35}
, {79, -87, 4}
, {-83, 18, -94}
, {-8, 18, 13}
, {34, 50, 24}
, {26, 67, -17}
, {67, 108, 25}
, {-132, 2, 106}
, {-57, 79, -36}
, {70, 25, 33}
, {26, -39, -32}
, {12, 10, -76}
, {-2, 9, 38}
, {-51, -27, 59}
}
, {{-127, -51, -107}
, {97, -74, -76}
, {43, 83, 69}
, {-82, -83, -13}
, {103, 101, 154}
, {120, -21, 24}
, {-12, 105, 87}
, {67, 128, 10}
, {26, -5, -79}
, {-61, 4, 81}
, {53, 5, 0}
, {0, -98, 3}
, {51, -69, -51}
, {7, 12, 7}
, {-46, -40, -112}
, {-69, 53, 48}
, {-38, -40, 26}
, {18, 5, 37}
, {27, 14, -21}
, {-20, -12, 15}
}
, {{122, -5, 88}
, {-46, -34, -25}
, {4, 124, 45}
, {31, 70, 23}
, {65, 69, -78}
, {51, 76, -186}
, {-60, 29, 21}
, {24, 77, -91}
, {86, 53, -42}
, {105, -91, -22}
, {43, -91, -9}
, {57, 62, -56}
, {-66, 48, 52}
, {61, -57, -112}
, {-18, 57, 27}
, {24, 83, 52}
, {1, -24, 75}
, {29, 35, 57}
, {41, -63, -113}
, {-46, -80, -128}
}
, {{84, -85, 9}
, {-54, 17, 94}
, {-59, 49, 100}
, {65, -60, -13}
, {-24, 62, -42}
, {-97, 80, 113}
, {61, 90, -55}
, {-35, -7, 99}
, {73, 94, -74}
, {-45, -51, 49}
, {-85, 59, -102}
, {-99, -87, -16}
, {61, -27, 20}
, {0, -65, -2}
, {-59, 51, 38}
, {11, 7, 76}
, {58, 35, -41}
, {78, 34, -108}
, {-112, 11, -65}
, {-109, -89, 37}
}
, {{37, 111, 30}
, {47, -27, -1}
, {-72, -64, -52}
, {-21, 12, 101}
, {60, 59, 13}
, {-44, 12, -88}
, {13, -75, -10}
, {-16, 137, 99}
, {-53, 99, -31}
, {60, -38, 26}
, {-47, -57, -4}
, {15, 10, -51}
, {15, -30, 88}
, {22, 136, -110}
, {-5, -69, -53}
, {0, 56, 11}
, {101, -50, -58}
, {80, 7, -45}
, {-84, -8, 23}
, {66, -32, 3}
}
, {{44, 78, -67}
, {89, 88, -48}
, {124, -22, 23}
, {20, -44, -39}
, {58, 35, -32}
, {52, 79, -87}
, {117, 103, 23}
, {39, -100, -59}
, {116, 15, 85}
, {87, 73, 105}
, {64, -23, -4}
, {24, -44, 69}
, {-100, 11, -35}
, {-71, 58, -20}
, {-31, -21, -77}
, {-16, -97, 13}
, {-49, -93, 68}
, {62, -76, 67}
, {83, 24, 49}
, {90, 1, -67}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE