/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_136_bias[CONV_FILTERS] = {-26, -31, 58, -31, -12, -56, -20, -28, 15, -42, -20, 61, -33, -4, -48, -13, -29, -30, -27, -24, -41, -52, 66, -43, 99, -53, -34, 73, 62, 78, -88, -27, -38, 32, -20, -11, 48, -42, -58, 171, -57, -35, -19, -44, 112, -46, -3, -11, 152, -47, 19, -41, -42, 61, -23, -14, -30, 110, 90, 22, -17, -24, -27, 116}
;

const int16_t conv1d_136_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-26, -113, -66}
, {-91, -116, -39}
, {-83, 20, -70}
, {57, -11, -98}
, {-161, -77, -65}
, {-48, -17, -98}
, {-15, 106, -3}
, {-82, 175, 27}
, {-6, 91, -2}
, {-10, -11, -34}
, {-81, 71, -39}
, {-28, -12, 101}
, {-46, -60, -39}
, {24, 41, -16}
, {-8, -18, -124}
, {112, 87, -43}
, {-14, -21, 18}
, {-15, -60, 5}
, {-69, -26, -43}
, {73, 83, -33}
, {41, 10, 12}
, {57, 27, -27}
, {-21, -45, -66}
, {-16, -45, -59}
, {-40, 30, 18}
, {-71, 0, 36}
, {152, 54, -18}
, {-25, 50, -9}
, {92, 12, 34}
, {-64, 16, 85}
, {23, 32, -62}
, {-19, -43, -52}
}
, {{40, -24, -96}
, {-26, -76, -25}
, {16, -43, -71}
, {-73, -9, -77}
, {18, -91, -46}
, {-60, 39, -93}
, {-40, 43, 7}
, {-31, -80, -3}
, {21, -45, 66}
, {-87, 42, 10}
, {-14, 0, -96}
, {-37, -28, -75}
, {-63, -66, -55}
, {24, -99, -6}
, {-91, 36, -86}
, {48, -33, -70}
, {16, -24, 10}
, {6, -4, 14}
, {-74, -62, -1}
, {41, 29, -83}
, {-40, -24, -91}
, {-37, -65, -7}
, {-95, -25, -84}
, {10, -17, 33}
, {17, -44, 35}
, {46, 27, 15}
, {-71, -31, -28}
, {28, -99, -71}
, {7, -82, 5}
, {-50, -36, -99}
, {-91, -4, 38}
, {-58, -13, -62}
}
, {{-31, -50, -4}
, {-17, -26, -25}
, {22, -51, -58}
, {-102, -95, 14}
, {-9, -65, 79}
, {-44, -55, -89}
, {26, -35, 38}
, {-55, 18, -63}
, {27, -20, 6}
, {35, -39, 9}
, {-63, -89, -65}
, {-64, 13, -78}
, {-97, -17, -97}
, {-61, -66, -31}
, {0, -45, -89}
, {91, 95, 58}
, {-85, -102, 18}
, {40, -68, -17}
, {-78, -59, 9}
, {-26, 40, -33}
, {-94, -54, -5}
, {-80, -30, -49}
, {33, -88, -96}
, {10, 23, -27}
, {-67, -160, -103}
, {-100, 16, 15}
, {-30, -30, 20}
, {-54, 19, 4}
, {-97, -50, -56}
, {-66, -27, -27}
, {25, -44, 11}
, {31, -3, -62}
}
, {{-21, 31, -98}
, {-5, -7, -14}
, {16, 2, -28}
, {-87, -41, -98}
, {-60, 31, -48}
, {0, 32, -4}
, {-86, 5, -18}
, {-45, -15, -35}
, {-16, -18, 15}
, {-42, 11, -95}
, {15, -40, -22}
, {-2, 0, 18}
, {38, 0, 1}
, {-98, -102, -37}
, {36, -104, -47}
, {-5, 35, 6}
, {-39, -90, -60}
, {-55, -24, 19}
, {-16, -73, -78}
, {-21, -60, -83}
, {-90, 24, 17}
, {-70, -12, -62}
, {9, -54, -43}
, {-15, -8, -12}
, {-55, -40, -58}
, {-6, 42, -103}
, {-71, 4, -22}
, {-95, -37, -88}
, {36, -23, 35}
, {-104, -97, -52}
, {-102, -46, 25}
, {-65, -6, -78}
}
, {{85, 98, -8}
, {-23, -43, 4}
, {21, -30, -66}
, {38, -64, -35}
, {6, -39, 11}
, {-17, -55, -44}
, {10, -62, -24}
, {20, 37, 5}
, {8, 23, -55}
, {88, 143, 20}
, {-59, 89, 47}
, {54, 7, 33}
, {78, 95, -49}
, {29, 22, -55}
, {13, -1, 44}
, {84, 26, 28}
, {41, 51, 52}
, {9, -11, 6}
, {-22, -18, -5}
, {-86, -18, -69}
, {47, -93, 27}
, {12, -4, 25}
, {-93, 20, -93}
, {-54, 71, 62}
, {26, 111, 1}
, {-36, 24, -31}
, {-30, -1, -46}
, {99, -13, -44}
, {-88, -90, -78}
, {-95, 1, -123}
, {78, 6, -15}
, {-64, 43, 11}
}
, {{-1, -53, 24}
, {25, -106, -12}
, {46, -81, 35}
, {-56, 15, -110}
, {0, -47, -76}
, {42, 50, -60}
, {52, -10, -15}
, {80, 78, 61}
, {20, -18, -36}
, {5, -9, -45}
, {-122, -1, -102}
, {75, 8, 44}
, {48, -1, 68}
, {8, -27, 34}
, {-4, -79, 11}
, {-87, 15, 56}
, {48, -20, 47}
, {10, 31, 30}
, {-49, -27, -77}
, {-74, -52, 5}
, {47, 38, -27}
, {-106, -25, -23}
, {19, -75, -36}
, {32, 22, 24}
, {14, 4, 22}
, {-87, -53, -68}
, {102, 109, 54}
, {24, 23, 34}
, {-30, -76, -24}
, {26, -52, 31}
, {-11, 68, -24}
, {-17, -71, -79}
}
, {{8, 14, 34}
, {54, 58, 14}
, {39, 26, -14}
, {0, 33, -32}
, {75, 22, 6}
, {8, -38, 38}
, {40, -6, 41}
, {57, 23, 38}
, {-33, 22, -70}
, {-4, 63, 57}
, {-29, -37, 32}
, {45, -17, 20}
, {7, 3, 35}
, {-5, 43, 10}
, {36, -22, -56}
, {72, -24, -23}
, {5, 46, -23}
, {-44, 5, 95}
, {22, -18, -80}
, {-6, 14, 72}
, {-1, -7, -41}
, {13, -20, 46}
, {-26, -13, 45}
, {-20, 24, 43}
, {72, -16, 63}
, {37, -81, 8}
, {23, 38, 116}
, {-45, -1, 7}
, {26, -23, -49}
, {8, -5, 61}
, {-14, 36, -43}
, {51, 37, -7}
}
, {{-10, -72, -16}
, {-41, 30, -109}
, {-34, -94, 7}
, {-64, -77, -29}
, {28, -74, -53}
, {-44, -27, -92}
, {-71, -20, 41}
, {-73, -1, -69}
, {-62, 7, 39}
, {-67, -45, -13}
, {-55, -15, -4}
, {-6, -57, 15}
, {-101, -60, -27}
, {-30, -93, 3}
, {-82, 30, -70}
, {-17, 2, 35}
, {-8, -49, -49}
, {11, -19, -84}
, {8, -38, 21}
, {-41, -111, -45}
, {-2, -26, 5}
, {-37, 26, -35}
, {18, -32, -74}
, {-59, -37, -8}
, {-10, -34, 18}
, {-87, -28, -94}
, {-30, -14, -77}
, {-29, -38, -73}
, {-65, -14, -79}
, {15, -68, -63}
, {-35, 7, -97}
, {0, -99, -14}
}
, {{38, 44, -52}
, {-44, -3, 75}
, {32, 65, 16}
, {-38, -14, 87}
, {27, 92, -15}
, {-32, -42, -8}
, {-19, -28, 22}
, {-75, -113, -82}
, {17, -4, 36}
, {34, -133, -35}
, {15, 40, 20}
, {-43, -72, 45}
, {-17, -54, 13}
, {-16, -9, 17}
, {26, 1, 82}
, {-36, 42, -33}
, {-80, -77, -45}
, {-36, -9, -97}
, {124, 42, 55}
, {-23, -80, 2}
, {-21, -101, 5}
, {65, -47, -15}
, {81, 88, 28}
, {42, 8, -101}
, {35, -168, -25}
, {-13, -22, 15}
, {-148, -38, -88}
, {12, -11, 12}
, {21, 60, 95}
, {47, 56, 31}
, {33, 58, -35}
, {51, 46, 20}
}
, {{-42, -76, 4}
, {-79, -93, -80}
, {18, -45, 53}
, {7, -99, -44}
, {94, 11, 25}
, {55, 51, -82}
, {-42, 83, -24}
, {5, 99, -28}
, {-1, -16, -11}
, {70, 53, 22}
, {-55, 1, -54}
, {77, 43, 18}
, {-3, 8, 75}
, {14, 62, 26}
, {54, 33, 23}
, {17, -48, -49}
, {0, 35, 27}
, {43, 1, 74}
, {-2, 60, 32}
, {35, 2, 59}
, {-33, 66, 25}
, {-35, -60, -90}
, {-2, -64, 36}
, {-3, 71, -15}
, {70, -14, 101}
, {26, 23, 5}
, {40, 11, 0}
, {87, 75, 73}
, {11, -42, 2}
, {-20, -64, -10}
, {28, -16, -40}
, {4, -10, 8}
}
, {{-28, -72, -32}
, {31, 1, 9}
, {-16, -32, 24}
, {-10, -37, 5}
, {-94, -42, 23}
, {-63, -86, -89}
, {10, -19, -77}
, {13, -67, 40}
, {-31, 1, 48}
, {-69, -22, 8}
, {25, -6, -5}
, {5, 24, -20}
, {-9, -43, 38}
, {-48, 35, -10}
, {-20, 21, -89}
, {-31, -56, 19}
, {-68, -20, -57}
, {-2, -55, -12}
, {-95, 10, -12}
, {-41, -43, -69}
, {-43, 20, -75}
, {18, 4, -23}
, {6, -95, -88}
, {15, 20, 19}
, {18, 9, -86}
, {-85, 2, -45}
, {-86, -26, 26}
, {0, -74, -23}
, {-11, -53, -7}
, {-60, -96, -64}
, {41, 12, -28}
, {-41, -52, -88}
}
, {{52, -5, -30}
, {-80, -88, -30}
, {-40, -118, -20}
, {-5, -51, -30}
, {-25, -43, -52}
, {-45, -96, -26}
, {18, -5, -42}
, {1, -14, -49}
, {-59, 9, 22}
, {-28, 12, 41}
, {-1, 55, -46}
, {-14, -71, -63}
, {-11, 0, -90}
, {-72, 28, -89}
, {-79, 16, -52}
, {44, -36, -25}
, {-10, -33, -78}
, {-12, -8, 124}
, {56, -38, -17}
, {-42, -52, 31}
, {-89, -63, -31}
, {-80, -46, -31}
, {-67, -158, -67}
, {-56, -90, 11}
, {-41, 129, 83}
, {51, -34, -68}
, {126, 71, -108}
, {-18, -38, 6}
, {1, -48, -141}
, {77, 48, 61}
, {0, -67, -29}
, {44, -53, -64}
}
, {{-50, -39, -51}
, {-98, -93, -33}
, {-90, -22, -76}
, {-66, -91, -91}
, {-43, 11, 16}
, {25, 2, -80}
, {16, 2, 36}
, {30, -17, -77}
, {-24, -20, -71}
, {-24, 32, 6}
, {45, -56, 100}
, {-25, -65, -29}
, {5, -52, 22}
, {-31, 32, 29}
, {32, -62, 32}
, {-106, -53, -22}
, {-44, 0, 13}
, {88, 43, 33}
, {52, -60, -37}
, {-10, 1, -59}
, {27, -60, -42}
, {55, 111, -30}
, {73, 40, -67}
, {21, -31, -52}
, {-52, -6, -9}
, {-12, -74, 53}
, {60, -34, -35}
, {50, 40, 88}
, {-93, -26, 31}
, {-75, -28, -41}
, {-1, 31, 16}
, {88, 8, -36}
}
, {{13, -97, -38}
, {-9, -90, 10}
, {-67, -38, 16}
, {-61, -56, -25}
, {7, -19, -31}
, {43, 27, -19}
, {-40, 44, 9}
, {35, -32, 26}
, {2, 4, 30}
, {7, 15, 37}
, {8, -31, -78}
, {3, 13, 10}
, {-57, -100, 6}
, {-96, 26, 3}
, {39, 23, -52}
, {-56, -77, 0}
, {26, -1, -9}
, {-86, -36, -59}
, {-14, -11, 4}
, {44, -58, 46}
, {-19, 17, 0}
, {56, -20, -82}
, {-17, 1, -67}
, {14, -53, -76}
, {-61, -74, -22}
, {-79, -48, -87}
, {24, -71, 35}
, {-16, 14, 15}
, {-35, -91, 9}
, {-31, 14, -86}
, {-43, -33, 0}
, {-22, -48, -52}
}
, {{-59, -50, -69}
, {54, 30, -56}
, {43, 9, -101}
, {-84, -55, -69}
, {-75, 49, -58}
, {-102, -48, -3}
, {1, 69, -60}
, {23, 6, 66}
, {-32, -16, 57}
, {27, 30, -61}
, {-12, -106, 57}
, {55, -45, 0}
, {15, 9, -26}
, {-7, -61, -2}
, {-14, 6, -1}
, {-84, -25, 56}
, {-13, -20, 0}
, {94, -49, -14}
, {48, 3, -39}
, {10, -33, -93}
, {-85, -56, -34}
, {150, 16, 2}
, {48, -8, -51}
, {16, 40, -12}
, {17, -48, -27}
, {-39, -52, 12}
, {-6, 3, 13}
, {-37, 20, -8}
, {35, -87, -98}
, {18, -116, -122}
, {46, -24, 28}
, {-50, 49, -81}
}
, {{1, -7, -68}
, {-42, -4, 16}
, {12, -44, 27}
, {-14, -24, -73}
, {28, -37, -15}
, {-88, -5, 59}
, {56, -67, -66}
, {60, 37, -41}
, {-37, 24, -28}
, {63, 11, 21}
, {-58, 15, -63}
, {1, 47, 6}
, {96, -5, -21}
, {44, -48, 32}
, {-50, -79, -6}
, {-87, -14, 26}
, {-53, 3, 144}
, {13, -11, 84}
, {-54, -22, -108}
, {52, -1, 19}
, {-23, -58, -22}
, {-79, 25, -33}
, {29, -77, -95}
, {-46, -8, 36}
, {-4, 103, 82}
, {18, 24, 96}
, {87, 83, 33}
, {-46, -40, 35}
, {-36, -43, -96}
, {-6, -13, 101}
, {-40, 50, 41}
, {0, -4, -38}
}
, {{-82, -93, -102}
, {-58, -85, -99}
, {-81, 22, -56}
, {3, 29, -74}
, {42, -41, -59}
, {-1, -66, -87}
, {-14, -9, -21}
, {23, 56, -13}
, {-64, 4, 51}
, {11, 7, 34}
, {-56, -13, -7}
, {-74, -72, -97}
, {-88, -9, -9}
, {-50, 14, -3}
, {-23, 0, -75}
, {28, -85, -78}
, {-56, 0, -74}
, {13, 21, -58}
, {15, -52, -9}
, {-46, 8, -52}
, {-93, 24, 28}
, {-73, -64, -64}
, {-2, -104, 1}
, {15, 1, -64}
, {-31, -24, -35}
, {100, 28, 0}
, {20, -68, -2}
, {-96, 0, -19}
, {-17, -65, -51}
, {-106, 17, -112}
, {-6, -60, -36}
, {-39, 17, 3}
}
, {{-92, -1, -62}
, {-71, -37, -98}
, {-83, -2, -70}
, {-47, 15, -46}
, {15, -98, -80}
, {-99, -66, 14}
, {-101, 13, 6}
, {-54, 31, -32}
, {-54, -56, 4}
, {17, 23, -34}
, {-1, -3, -97}
, {-90, -56, 36}
, {-34, -28, -94}
, {-70, -83, -39}
, {-6, -66, -57}
, {72, 44, -31}
, {-69, -11, 34}
, {-40, -13, -15}
, {31, -45, 22}
, {-12, 37, -4}
, {-78, -28, 37}
, {7, 4, 41}
, {-57, -17, -66}
, {-18, 31, -42}
, {-39, -34, -44}
, {19, -97, -47}
, {-30, -83, -23}
, {13, -25, -50}
, {23, -64, -61}
, {-64, 11, -34}
, {-101, 14, -102}
, {-26, 8, 12}
}
, {{56, 5, 41}
, {3, -7, -12}
, {-88, 41, 47}
, {-3, -22, -21}
, {81, -50, -52}
, {-6, -65, -68}
, {87, -19, 31}
, {57, -15, 84}
, {-39, -21, 65}
, {36, 121, 16}
, {50, 50, -55}
, {46, -25, 71}
, {-28, 4, 10}
, {54, 23, 41}
, {0, -41, -81}
, {63, -50, -24}
, {-30, -51, -16}
, {14, 30, 33}
, {16, 51, -17}
, {-2, -3, -2}
, {-50, -62, 25}
, {-49, 63, -63}
, {9, 13, -42}
, {-10, 37, -38}
, {-35, 38, -47}
, {-17, 1, 52}
, {-14, 61, 7}
, {-30, 17, -36}
, {-31, -34, 4}
, {-47, -33, -82}
, {29, 20, 78}
, {57, -13, -76}
}
, {{85, -42, 11}
, {-50, 9, -62}
, {68, 21, -33}
, {6, 14, 0}
, {-9, -21, -16}
, {13, 15, 36}
, {65, 71, 39}
, {-27, -13, -21}
, {4, 55, -13}
, {35, 52, -86}
, {54, 35, -81}
, {20, -45, -49}
, {68, 11, 7}
, {-21, 19, -12}
, {64, -5, -48}
, {98, -20, -84}
, {-46, -25, -20}
, {-18, -16, -60}
, {41, -35, -42}
, {45, 56, 70}
, {20, 21, -38}
, {20, -6, 53}
, {86, 77, -97}
, {7, 75, 17}
, {-7, -9, -1}
, {16, -18, 51}
, {-28, -27, 31}
, {58, 36, 49}
, {76, 12, 102}
, {55, 13, 1}
, {123, -8, 7}
, {54, 37, 38}
}
, {{-8, -102, -10}
, {14, -72, -19}
, {-65, 40, 23}
, {-67, 8, -92}
, {-12, -12, -115}
, {-44, 12, -5}
, {24, -5, -19}
, {-42, -8, -38}
, {-19, -18, 47}
, {79, 5, -28}
, {1, -59, 46}
, {-73, -87, -18}
, {-62, -29, 67}
, {-47, 16, 15}
, {-67, -62, -33}
, {1, 17, -34}
, {60, 89, 36}
, {-60, -5, 35}
, {20, -4, 21}
, {29, 46, -37}
, {-11, 80, 47}
, {-59, 85, 107}
, {-69, -14, 42}
, {15, 40, 90}
, {101, 48, 42}
, {-15, -57, 3}
, {-30, 30, 12}
, {50, -17, -17}
, {39, 29, -90}
, {6, -21, 79}
, {-114, 7, -48}
, {-102, -42, 47}
}
, {{-80, -32, -126}
, {-52, 11, 4}
, {-46, -46, -19}
, {6, -60, -15}
, {-109, 27, -29}
, {-34, -66, 42}
, {-63, -39, -29}
, {-52, -7, 6}
, {-54, -21, 63}
, {-32, 12, -60}
, {-33, 0, 3}
, {-54, -1, 35}
, {-23, -2, -42}
, {-71, 15, -16}
, {32, -36, 44}
, {45, 84, -49}
, {2, 34, -22}
, {94, 30, -35}
, {-71, 0, -4}
, {5, -58, -58}
, {-8, 31, 72}
, {103, -29, -75}
, {27, -102, 41}
, {55, -13, -69}
, {-56, 60, -7}
, {15, -8, -72}
, {61, 16, -73}
, {-25, -29, 61}
, {38, -29, -27}
, {-18, -76, -18}
, {-34, -12, -95}
, {26, -5, 7}
}
, {{-2, -134, -122}
, {-110, -65, -61}
, {-77, -13, 71}
, {-108, -58, -174}
, {28, 69, 48}
, {-24, 53, -42}
, {-44, -94, 49}
, {-44, -18, -87}
, {59, 11, -61}
, {13, -88, -117}
, {75, 150, -93}
, {-39, -39, 45}
, {59, -26, -83}
, {-29, 10, -94}
, {-167, 80, 13}
, {-47, -41, -68}
, {-94, -134, 102}
, {-8, 11, 159}
, {-92, 31, -112}
, {34, 25, -48}
, {-88, 103, 40}
, {-31, 101, 61}
, {-11, -82, 14}
, {30, -12, 89}
, {54, 96, 26}
, {0, 31, -23}
, {64, 22, 33}
, {20, -53, -37}
, {-48, -70, -58}
, {-24, 6, 225}
, {-66, -76, -138}
, {14, 16, -21}
}
, {{11, 23, -51}
, {-50, 25, -51}
, {53, -97, -94}
, {52, 107, 124}
, {40, 0, 59}
, {6, -34, 66}
, {33, 12, -60}
, {41, -7, -96}
, {20, -62, 11}
, {30, -66, 57}
, {-42, -23, 33}
, {30, -96, -17}
, {2, -80, -77}
, {43, -51, -86}
, {11, -16, -61}
, {-36, 1, 16}
, {-73, -28, -41}
, {7, -135, -5}
, {97, -99, 28}
, {-97, -28, 52}
, {70, -79, 29}
, {52, 36, 19}
, {2, 38, 69}
, {-37, -72, 16}
, {1, -2, -58}
, {14, -90, 41}
, {-43, 23, -169}
, {-35, 10, -31}
, {-98, -71, -11}
, {-88, 40, 42}
, {91, -34, -107}
, {102, -9, 34}
}
, {{3, 60, -79}
, {-14, -2, 20}
, {-59, 32, -17}
, {-49, 6, 51}
, {-27, -101, -63}
, {36, 61, -21}
, {-2, -11, -30}
, {52, 32, 34}
, {45, -3, 34}
, {-63, -91, -92}
, {30, -42, -30}
, {42, 23, 0}
, {-84, -12, -37}
, {-17, -35, 35}
, {58, -49, -27}
, {56, 74, 72}
, {-54, -91, 27}
, {-43, 0, -9}
, {41, -4, -27}
, {-19, 28, -36}
, {-70, -1, -26}
, {-27, -91, -56}
, {46, -5, 53}
, {-29, 39, -91}
, {-86, -33, -11}
, {47, 70, -29}
, {-13, -85, 29}
, {-85, -92, 22}
, {24, -49, -75}
, {32, -69, -89}
, {38, -92, -90}
, {28, -7, -46}
}
, {{-77, 57, 92}
, {30, 32, 16}
, {-55, -3, -75}
, {-102, 44, 67}
, {-70, -62, -24}
, {-49, 35, -14}
, {-19, 61, -28}
, {-66, -21, -3}
, {-99, 62, -50}
, {-7, -75, 79}
, {-51, 76, 81}
, {-55, -41, -68}
, {-51, -5, 30}
, {-93, -50, 0}
, {39, -101, -61}
, {-69, -30, 0}
, {39, -58, 5}
, {-30, -9, 32}
, {11, 26, -75}
, {-1, -39, 40}
, {-93, -60, -82}
, {35, -117, -6}
, {-50, 22, -15}
, {-59, -104, -87}
, {-11, -36, 22}
, {-36, -66, 12}
, {46, -22, -2}
, {-89, -49, -20}
, {15, -92, -1}
, {37, 3, -55}
, {-40, 62, 65}
, {-31, 22, -35}
}
, {{-35, 8, 27}
, {-78, 18, -4}
, {-73, -86, -85}
, {10, -101, -86}
, {-87, -38, -20}
, {-72, 3, -69}
, {-68, -103, 19}
, {-68, -25, -54}
, {-7, -23, 9}
, {-62, -20, -4}
, {-37, 11, 0}
, {-65, -74, -10}
, {-20, -70, 32}
, {-92, -49, -42}
, {-9, -18, 68}
, {-30, -5, -61}
, {-61, -7, -95}
, {23, -99, 19}
, {-1, -84, -23}
, {31, -82, -31}
, {-83, 21, -96}
, {-40, -102, -15}
, {-98, -41, -6}
, {-18, -40, 18}
, {-3, -53, -21}
, {-49, 24, -33}
, {10, -47, -42}
, {29, 23, 13}
, {-7, -3, 42}
, {-27, -60, -22}
, {-83, -21, -72}
, {-51, 41, -13}
}
, {{8, -53, -34}
, {48, -33, -58}
, {-56, -93, 15}
, {-59, -32, 43}
, {51, -85, -26}
, {6, 91, -18}
, {21, 15, -82}
, {-38, -57, 24}
, {3, -27, 22}
, {-93, -12, 34}
, {-26, -26, 70}
, {41, 28, -96}
, {-77, -33, -97}
, {2, -15, -3}
, {-13, -90, -2}
, {-91, 47, 92}
, {-35, -65, -52}
, {118, -41, -69}
, {41, 29, -26}
, {-38, -72, -14}
, {74, -118, 0}
, {39, 126, -132}
, {-35, -20, 74}
, {-8, 1, -68}
, {33, 20, -8}
, {-4, -94, 12}
, {31, -3, -13}
, {-44, -50, -81}
, {-20, 33, -79}
, {-6, 17, -62}
, {-48, -24, -48}
, {111, 8, -92}
}
, {{25, -104, -1}
, {52, 10, 40}
, {37, -53, 75}
, {-94, 10, 8}
, {-66, 60, 26}
, {-62, -51, -47}
, {26, -45, -57}
, {-109, -43, -51}
, {60, 26, 15}
, {-52, -40, -90}
, {31, 49, -67}
, {-7, -2, 32}
, {-11, -27, 44}
, {-45, 45, -30}
, {77, 80, 116}
, {9, -63, -54}
, {-75, -32, 69}
, {-87, -96, -19}
, {-58, -46, 29}
, {29, 13, -39}
, {8, 42, 0}
, {-13, 37, 85}
, {-22, 2, -55}
, {-12, 32, -18}
, {-56, -38, -32}
, {13, -50, 66}
, {-14, -84, -34}
, {43, 26, 19}
, {-44, 8, 61}
, {31, 7, 21}
, {-17, -14, -23}
, {-97, -15, 44}
}
, {{34, 23, -5}
, {-16, -44, 32}
, {64, -73, 82}
, {-78, -31, -59}
, {-86, 32, 34}
, {-10, 4, -12}
, {-1, -25, 40}
, {-2, -100, 32}
, {-69, -31, -70}
, {-37, 24, -21}
, {15, 95, 72}
, {-100, -55, 51}
, {6, -20, 56}
, {39, 49, 33}
, {-11, 14, 84}
, {-36, -81, 29}
, {-72, -70, 20}
, {-83, -1, -9}
, {-20, 17, 16}
, {38, -7, -39}
, {51, -85, -66}
, {-34, -18, 0}
, {16, 46, 62}
, {9, -45, -37}
, {-17, -95, -26}
, {4, -16, 26}
, {6, -34, -6}
, {-60, -95, -15}
, {29, -6, -13}
, {-54, -2, -2}
, {54, -16, 3}
, {-52, 63, -70}
}
, {{-133, -79, 0}
, {12, -4, 0}
, {-60, -56, 32}
, {-49, -117, -42}
, {12, -4, 54}
, {-94, -71, 16}
, {16, -109, 14}
, {-72, 105, -112}
, {4, -9, 71}
, {-148, -42, 25}
, {7, -125, -35}
, {-68, -87, 75}
, {-35, -28, 58}
, {-5, -84, -24}
, {-58, 118, 149}
, {-136, -31, -48}
, {-79, 69, 269}
, {-51, -51, 137}
, {0, -26, 11}
, {-69, -8, -48}
, {-75, 44, -9}
, {-37, 27, -12}
, {-1, -39, -73}
, {-95, 47, 51}
, {100, -83, -148}
, {-91, -51, 18}
, {-75, -143, 35}
, {-24, 27, 57}
, {-126, -100, 38}
, {-112, -117, 48}
, {-79, -42, 42}
, {-59, -80, -67}
}
, {{-92, -40, -21}
, {23, -16, -42}
, {-46, -5, -66}
, {-94, -5, -40}
, {-6, 11, 16}
, {-15, 42, -81}
, {-79, -35, 22}
, {-37, -9, -37}
, {-3, 53, -62}
, {-101, -70, -50}
, {-63, -28, -18}
, {-33, -97, 22}
, {-86, -45, -39}
, {-23, -30, 28}
, {-21, -45, 28}
, {89, -89, 43}
, {-12, -86, -89}
, {23, -10, -3}
, {9, -95, 13}
, {-25, -23, 14}
, {-95, -35, 16}
, {-11, -43, -32}
, {28, -89, -88}
, {-22, -87, 39}
, {43, -20, 8}
, {-72, -51, 29}
, {-17, -20, 35}
, {-35, -101, -92}
, {39, -57, 31}
, {-62, -83, -63}
, {18, -95, 15}
, {-42, -82, -7}
}
, {{39, -23, -6}
, {-82, -25, -61}
, {-89, 12, -29}
, {12, 39, -100}
, {50, -36, -22}
, {24, -88, -20}
, {29, 70, -84}
, {2, 64, 47}
, {6, -73, 28}
, {-31, -81, -42}
, {-51, -47, -41}
, {-14, -40, -23}
, {57, 53, -72}
, {-32, -57, -73}
, {-9, -81, 17}
, {-43, -69, 53}
, {77, 39, 54}
, {28, -39, -79}
, {-8, -82, 31}
, {-83, -19, -28}
, {-74, -58, -83}
, {-32, -28, 15}
, {43, -9, -60}
, {-8, -26, -79}
, {28, 9, 28}
, {39, 7, -45}
, {-45, 51, -93}
, {-4, -69, 28}
, {-32, -43, 0}
, {-71, -44, -90}
, {-30, 103, 8}
, {-45, -64, -49}
}
, {{12, -121, -88}
, {8, -13, 41}
, {-84, -17, -79}
, {-59, 7, -4}
, {-40, 73, -85}
, {-93, -12, -10}
, {-135, 48, 68}
, {-23, 25, -44}
, {0, -52, -22}
, {48, 26, -133}
, {20, 3, 53}
, {-64, -75, 18}
, {-6, -7, -40}
, {-82, -62, 33}
, {-39, -47, -62}
, {34, 39, 49}
, {-82, 51, -141}
, {74, 39, 31}
, {97, 56, -87}
, {35, -95, 67}
, {-38, -73, 20}
, {31, -40, -76}
, {37, -78, -64}
, {41, 42, 31}
, {-89, 162, 30}
, {5, -5, 50}
, {72, -28, -105}
, {-37, 36, -33}
, {-25, -12, -113}
, {-51, -7, 121}
, {-42, 55, -38}
, {81, -15, 34}
}
, {{48, 60, 28}
, {68, 26, 35}
, {91, -16, 36}
, {29, 62, -71}
, {32, -8, -12}
, {-34, 32, 50}
, {27, 37, -1}
, {118, -29, 39}
, {76, -72, -13}
, {31, 54, 8}
, {55, 4, 34}
, {32, 84, 56}
, {14, -44, -16}
, {42, -44, 70}
, {-48, 33, -42}
, {22, -13, 89}
, {-15, -5, 31}
, {-38, 47, -25}
, {0, 32, -17}
, {50, 81, -7}
, {46, 27, 7}
, {22, -36, 62}
, {15, -3, -31}
, {61, 24, -33}
, {-4, 69, 18}
, {-18, -60, 28}
, {-38, 69, 105}
, {89, 75, 88}
, {31, 24, -33}
, {6, 48, -2}
, {93, -39, -65}
, {-7, 43, -17}
}
, {{34, 31, -18}
, {23, 9, -31}
, {-22, -36, 44}
, {43, 48, 59}
, {116, 34, 29}
, {69, 24, -17}
, {75, -28, 38}
, {-33, -24, -16}
, {-18, -6, -54}
, {64, 25, 67}
, {-71, 22, 68}
, {48, 45, 89}
, {6, 41, 51}
, {73, 21, 65}
, {-29, -68, 41}
, {94, 87, 17}
, {-6, 39, 68}
, {20, 5, 23}
, {9, 12, -14}
, {25, 1, -17}
, {18, -45, 16}
, {66, 86, -88}
, {8, 41, -24}
, {38, 87, 38}
, {65, 93, -25}
, {-34, 9, 0}
, {62, -31, -11}
, {83, -16, 5}
, {9, -48, -59}
, {-18, -90, -59}
, {-21, 23, 78}
, {90, 26, 0}
}
, {{9, -26, -26}
, {-24, 34, 42}
, {85, 9, 79}
, {46, 55, -39}
, {22, -30, 84}
, {83, 34, 71}
, {76, -43, 3}
, {-96, -32, 85}
, {2, -75, 32}
, {23, -115, -94}
, {15, 23, 32}
, {-10, -36, -69}
, {64, 94, 44}
, {-38, 33, -56}
, {93, 103, 173}
, {-46, -25, -66}
, {-12, -41, 110}
, {-16, -28, -11}
, {63, 38, 7}
, {16, -82, -32}
, {12, 37, 46}
, {-36, -60, 72}
, {52, 63, 13}
, {63, 59, -8}
, {-63, -23, 95}
, {-46, 60, 3}
, {-23, -13, -14}
, {26, -20, -54}
, {23, -49, 32}
, {16, 7, -66}
, {152, -35, 18}
, {67, 30, -60}
}
, {{-87, -58, 3}
, {25, -45, 17}
, {-86, 2, -31}
, {-19, -131, -72}
, {-25, 3, 21}
, {44, -39, -54}
, {-55, -13, -36}
, {-48, 39, 25}
, {-61, -40, -56}
, {-71, -70, -6}
, {68, -115, -8}
, {82, 54, -43}
, {-57, 6, -40}
, {-45, 50, -92}
, {34, 37, -10}
, {-102, 4, -2}
, {48, 96, -22}
, {-37, -26, 4}
, {-62, 28, 8}
, {-70, -6, -24}
, {-25, -14, 3}
, {-81, -62, 38}
, {58, 22, -39}
, {15, -13, 19}
, {84, 14, 44}
, {-27, -15, -23}
, {63, 54, 33}
, {11, -2, 78}
, {42, 29, 40}
, {-72, -44, -87}
, {-43, 37, 12}
, {30, 46, -80}
}
, {{-77, -43, -46}
, {9, -93, -34}
, {-63, -57, 30}
, {-28, -39, 0}
, {-64, -90, 29}
, {-46, -27, -29}
, {25, -22, 0}
, {97, -19, 89}
, {17, 9, -51}
, {-68, 51, 9}
, {14, -44, -27}
, {-55, 24, 19}
, {-73, -108, -21}
, {-3, -52, 6}
, {10, -41, -95}
, {20, 59, 56}
, {-17, -47, -71}
, {-34, -2, -20}
, {18, -31, -106}
, {63, -30, -26}
, {74, 44, 56}
, {24, 33, -21}
, {20, -81, -81}
, {40, -38, -16}
, {45, -34, 106}
, {-40, -51, -80}
, {-65, -81, 5}
, {-38, 48, -75}
, {-42, 14, -10}
, {0, -2, -70}
, {23, -13, -83}
, {-90, -20, -29}
}
, {{28, -54, -49}
, {-84, -79, -90}
, {-45, -17, 32}
, {-4, -62, -46}
, {-17, 31, -31}
, {-84, 14, -44}
, {-50, -50, 20}
, {-103, -3, -43}
, {-57, 70, -15}
, {37, 35, 30}
, {0, -7, -8}
, {-11, 35, 15}
, {-98, 0, -64}
, {-35, -91, -68}
, {40, -93, -63}
, {76, 141, 173}
, {-20, -72, 37}
, {-28, -13, -85}
, {-69, 0, 18}
, {-57, -31, -99}
, {-59, -78, -3}
, {-64, -63, 22}
, {-45, 21, 15}
, {-12, -104, -48}
, {23, -78, -20}
, {16, -6, 31}
, {-62, 25, -68}
, {-95, -36, -100}
, {53, -84, -93}
, {78, -9, -83}
, {-92, -13, -5}
, {-83, -77, -58}
}
, {{5, -20, -27}
, {-9, -104, -4}
, {-37, -26, -22}
, {-94, 12, -12}
, {-5, -42, -117}
, {-100, -117, -76}
, {-52, -70, -47}
, {1, 4, 22}
, {-72, 46, -55}
, {-8, -42, 10}
, {-17, 42, -13}
, {-43, -92, -51}
, {-88, -10, -89}
, {-26, -65, -64}
, {-52, 0, 19}
, {-11, 13, -41}
, {-90, -12, -89}
, {-56, -46, 7}
, {-56, 1, -86}
, {-106, -16, 14}
, {5, -6, -42}
, {-73, -124, -3}
, {-40, 13, 8}
, {25, -64, -47}
, {67, -89, -37}
, {26, -30, 2}
, {-98, -71, -97}
, {17, -81, -79}
, {0, -78, -46}
, {-67, -96, -66}
, {-14, 23, -81}
, {-65, -25, -41}
}
, {{-69, 49, -16}
, {54, 4, -36}
, {-15, -79, -21}
, {-18, -70, -48}
, {-4, 8, 15}
, {-15, 10, 17}
, {-18, -26, -11}
, {-93, -6, -104}
, {45, -67, 18}
, {0, -68, 32}
, {10, 0, 23}
, {26, -89, -44}
, {-101, -12, -52}
, {-31, 21, -70}
, {-26, -58, -51}
, {-86, -20, 9}
, {-90, -32, -19}
, {-94, 3, -64}
, {-87, 10, 19}
, {-92, -50, -91}
, {19, 18, 27}
, {-54, -70, 40}
, {-9, -80, -60}
, {0, 28, -54}
, {-8, -73, -24}
, {-53, 1, -20}
, {38, -75, -91}
, {0, 27, 36}
, {-65, -47, 2}
, {-7, 16, -78}
, {1, 2, -87}
, {-36, -66, -97}
}
, {{-32, 35, 49}
, {-26, 16, 5}
, {53, -3, 54}
, {75, 42, -68}
, {17, 31, -35}
, {50, 63, -87}
, {39, -57, 18}
, {73, 71, 23}
, {47, 20, 5}
, {87, -9, -25}
, {70, 72, 10}
, {22, 78, 52}
, {60, -25, 24}
, {68, -59, -51}
, {-6, 18, -12}
, {65, -44, -3}
, {31, -8, 93}
, {88, -12, -46}
, {-30, -74, -47}
, {71, -3, 6}
, {94, 65, 44}
, {54, 39, -64}
, {3, 45, 50}
, {5, 40, 75}
, {-15, 21, 47}
, {-44, 53, -30}
, {52, -24, 30}
, {-30, -6, -23}
, {-57, -14, 25}
, {-48, -101, -93}
, {68, 73, -37}
, {-42, 17, -64}
}
, {{-16, 10, 96}
, {10, 67, 58}
, {67, 41, 2}
, {-69, 52, -9}
, {50, 66, 82}
, {-34, 51, 28}
, {-52, 118, -2}
, {93, -2, 18}
, {5, -102, 66}
, {80, 4, 106}
, {43, 30, 53}
, {-30, -2, 30}
, {16, 18, 30}
, {19, -44, -40}
, {5, 93, 19}
, {-85, -30, -14}
, {36, 91, -43}
, {-50, 35, -49}
, {108, 90, -19}
, {12, 21, 48}
, {-84, 0, 30}
, {-61, 36, -80}
, {-52, 81, 16}
, {-11, 33, -32}
, {25, -91, 107}
, {-53, 7, 5}
, {6, 47, 80}
, {7, 69, -22}
, {-64, -14, 57}
, {31, 67, -93}
, {105, 144, 122}
, {-50, 78, 5}
}
, {{1, -79, 40}
, {-39, -4, -71}
, {-55, -6, -34}
, {-87, -98, -28}
, {-39, -27, -61}
, {-8, 36, -27}
, {27, -61, -64}
, {6, -93, -29}
, {-32, 16, -46}
, {15, -74, 67}
, {-10, 0, 82}
, {-74, 30, -19}
, {-93, 2, -80}
, {40, 12, -96}
, {-38, -54, -70}
, {79, 138, 175}
, {32, -105, 9}
, {-90, -85, -92}
, {-88, 5, -81}
, {-81, 20, -5}
, {-26, -89, -78}
, {42, -18, -80}
, {-45, -62, 1}
, {-77, -30, 7}
, {-118, -7, -50}
, {-16, -70, -61}
, {-73, -48, 5}
, {-76, 1, 28}
, {6, 6, -70}
, {-24, 39, -107}
, {-36, -92, -32}
, {-76, -40, -49}
}
, {{-67, -65, -28}
, {-44, -36, 2}
, {-41, -77, -102}
, {-71, 17, -60}
, {30, 1, -60}
, {-29, 47, -42}
, {-38, 25, -54}
, {-40, 37, 63}
, {-39, 4, 6}
, {-98, -72, -76}
, {20, 71, -34}
, {9, 14, 9}
, {-73, -17, -50}
, {12, -68, -96}
, {-39, -100, -64}
, {23, 14, 52}
, {-79, -32, 38}
, {-10, -8, 58}
, {-21, -27, -3}
, {30, -100, -20}
, {-18, -51, 7}
, {-16, -41, 36}
, {-14, -8, -37}
, {38, 32, 25}
, {100, 109, -23}
, {-40, 22, 60}
, {29, 23, 5}
, {20, -51, 61}
, {-76, -49, 46}
, {-7, 36, -45}
, {-18, -79, 68}
, {45, 13, -74}
}
, {{0, -54, 24}
, {45, 45, 42}
, {8, 39, -28}
, {-17, -66, -83}
, {5, -13, 63}
, {0, -59, -52}
, {-47, 63, 34}
, {89, 25, 0}
, {1, -21, -2}
, {-35, 50, -73}
, {12, -33, -9}
, {42, -16, 40}
, {-106, 64, 17}
, {-39, -16, -58}
, {70, 45, -33}
, {75, 31, 46}
, {142, 65, 65}
, {19, -76, 108}
, {51, 55, -76}
, {1, -93, 74}
, {80, 71, -86}
, {-17, -37, 54}
, {-83, -27, 11}
, {-26, 0, -40}
, {17, 16, 76}
, {-89, -11, 26}
, {-47, -27, 7}
, {40, 75, 75}
, {-79, -72, 48}
, {-121, -26, -133}
, {30, -49, 9}
, {-37, -64, -72}
}
, {{49, 60, 48}
, {66, 10, -60}
, {-39, -53, 41}
, {101, 50, 13}
, {102, 94, -46}
, {-26, 55, 32}
, {79, 54, 17}
, {15, 70, -43}
, {-37, 7, -20}
, {-6, 74, -7}
, {-16, -32, 57}
, {58, 77, 17}
, {-56, 59, -69}
, {-54, 24, -62}
, {-41, -85, -43}
, {-36, 29, -35}
, {-35, -25, 35}
, {52, 89, 55}
, {57, -38, 8}
, {76, -62, -12}
, {84, 68, 22}
, {-22, 36, -21}
, {12, 0, 59}
, {82, 12, -19}
, {-33, -22, 40}
, {-13, -6, 16}
, {-24, 43, -23}
, {-20, 29, 76}
, {35, -35, 64}
, {3, -95, -105}
, {10, -5, -20}
, {51, 74, 37}
}
, {{-12, 89, 0}
, {-89, -51, -62}
, {-56, -21, -28}
, {-7, 3, -52}
, {48, 99, -9}
, {-93, -74, 45}
, {-4, 54, -26}
, {35, 5, -98}
, {5, -38, 67}
, {-15, -67, 0}
, {55, 67, -36}
, {7, -103, 24}
, {-87, -88, 20}
, {9, -14, -33}
, {-34, -29, -63}
, {56, 99, 228}
, {77, 45, 105}
, {-74, -14, -84}
, {-5, -5, -27}
, {-53, -86, -68}
, {-84, -55, 32}
, {16, 6, 34}
, {-76, 8, -79}
, {-27, 2, 1}
, {81, -31, -38}
, {-57, -36, -12}
, {0, 39, 34}
, {17, 4, -48}
, {45, 40, -26}
, {-36, -7, -158}
, {-116, -90, 0}
, {-72, -110, 23}
}
, {{14, -47, 39}
, {-2, -87, -99}
, {-49, -70, -39}
, {-12, -7, 7}
, {15, 51, 17}
, {20, 13, -1}
, {39, -71, 9}
, {3, 5, -82}
, {-55, 57, 23}
, {-28, -55, -78}
, {52, 46, -19}
, {-98, -74, -112}
, {-31, -87, -33}
, {32, -12, -88}
, {38, -1, -103}
, {-33, -54, -20}
, {-11, -133, 0}
, {-39, 3, 42}
, {12, -56, -47}
, {-6, -69, 9}
, {-111, -78, 35}
, {19, -28, 37}
, {-28, -45, -1}
, {-15, -31, -7}
, {8, -10, -79}
, {1, 30, 5}
, {-32, -172, -33}
, {24, -77, -69}
, {41, -53, 89}
, {-52, 73, 46}
, {37, -41, 8}
, {-27, -2, 11}
}
, {{15, 65, -41}
, {11, 48, 19}
, {-59, 34, -76}
, {46, -71, 4}
, {-23, 75, 56}
, {-15, -10, 27}
, {-31, -33, -24}
, {2, 81, -101}
, {-52, 21, -61}
, {75, 85, -38}
, {-73, -41, 22}
, {67, 93, -72}
, {60, 57, -40}
, {16, -77, -76}
, {-73, -20, -81}
, {64, -41, 28}
, {47, 84, 92}
, {-4, 52, 38}
, {-86, 19, 34}
, {-43, -31, -11}
, {-64, -15, 64}
, {6, -35, -62}
, {-51, -25, -39}
, {-6, 56, 51}
, {60, 56, 41}
, {107, 46, -16}
, {-23, -44, 33}
, {38, 80, 49}
, {-49, 11, -64}
, {-110, -112, -125}
, {28, -27, 42}
, {5, -12, -83}
}
, {{13, -26, 15}
, {-59, -86, 31}
, {-56, -101, -81}
, {-20, -5, -104}
, {-31, 46, 27}
, {-38, 73, -93}
, {64, -105, 21}
, {37, -3, 22}
, {2, 71, -25}
, {11, 7, -105}
, {40, 24, -31}
, {-80, -76, -101}
, {-12, 28, -78}
, {-92, -48, -9}
, {21, 18, -28}
, {16, 32, -7}
, {-89, 24, -63}
, {-102, -38, 40}
, {-97, -39, -8}
, {-58, -87, -4}
, {-54, 9, 10}
, {15, 67, -71}
, {-40, -82, 33}
, {-60, -50, 21}
, {21, 27, 26}
, {-93, -85, 0}
, {-91, -3, -22}
, {27, -64, 15}
, {-56, -11, 19}
, {-106, -90, -88}
, {-43, -80, -32}
, {-86, -17, -48}
}
, {{-38, -36, 53}
, {11, 11, -26}
, {60, 14, -24}
, {31, 29, -67}
, {-73, 75, -11}
, {-66, -36, -17}
, {36, 23, -72}
, {-31, -25, -16}
, {-48, -23, -23}
, {-6, -58, -90}
, {3, -31, 27}
, {42, -50, -32}
, {-85, 23, 17}
, {-17, -42, 35}
, {29, -6, -50}
, {-20, 3, 77}
, {-51, -105, -111}
, {-32, -58, -103}
, {48, 2, -42}
, {22, -90, -12}
, {86, 34, 77}
, {73, -18, 6}
, {44, -31, -70}
, {-74, -4, -50}
, {-40, 70, 29}
, {-81, -72, -66}
, {-92, -36, 34}
, {-63, -72, -36}
, {-43, -91, -85}
, {-5, -30, -69}
, {43, 9, -26}
, {-17, -104, -58}
}
, {{29, -31, -12}
, {38, -39, -79}
, {-83, -29, 12}
, {-22, 23, -13}
, {44, 27, -31}
, {-13, -59, -5}
, {-49, -90, -13}
, {-62, -64, -44}
, {-29, 12, 45}
, {-131, 126, 48}
, {-53, -97, -91}
, {9, -50, -14}
, {-100, -91, -74}
, {0, 0, 33}
, {-94, -14, 24}
, {7, -14, -18}
, {72, -33, -27}
, {-86, -70, 5}
, {6, -64, -61}
, {19, -38, -25}
, {-54, 6, 34}
, {-86, 31, -14}
, {-14, -67, 27}
, {-60, 16, -95}
, {10, 54, -124}
, {-108, -9, 10}
, {28, -96, 19}
, {-65, -77, -55}
, {-23, 43, -95}
, {16, 51, -94}
, {-113, 62, 11}
, {18, -39, -89}
}
, {{-71, -1, -23}
, {-43, -53, 37}
, {27, 18, 14}
, {-77, -83, -81}
, {22, 7, -26}
, {-63, -65, 37}
, {-101, 8, -15}
, {24, 4, -82}
, {6, 60, 47}
, {1, -72, 21}
, {0, -60, -23}
, {-49, -22, -26}
, {25, 27, -22}
, {-29, -37, -99}
, {-60, 49, 23}
, {-13, 24, -48}
, {-36, -40, -53}
, {-76, -71, 27}
, {-21, -61, -68}
, {-8, 25, -66}
, {-19, -99, -68}
, {-83, -31, 1}
, {19, 9, -23}
, {30, 7, -42}
, {-42, 25, -26}
, {-8, -90, 20}
, {-15, -47, -55}
, {-13, -2, 4}
, {19, -91, 20}
, {-12, -66, -96}
, {-95, -30, -98}
, {40, -94, -47}
}
, {{-35, 65, -62}
, {37, 14, 38}
, {35, -41, 55}
, {81, -37, -56}
, {98, 44, 57}
, {-11, 51, -91}
, {1, 43, 31}
, {1, -28, -15}
, {-13, 101, 65}
, {111, 83, 72}
, {20, -65, -62}
, {-29, 0, -41}
, {61, -73, 19}
, {25, -47, 25}
, {-25, 37, 19}
, {59, 5, 26}
, {7, 40, 71}
, {50, 22, 15}
, {-5, 65, 53}
, {50, -8, -21}
, {79, -39, 63}
, {-23, 34, 51}
, {60, 45, -42}
, {-8, 86, -22}
, {33, 38, 47}
, {1, 47, -19}
, {-30, 23, 20}
, {-4, 15, -30}
, {-30, -2, 59}
, {-58, -67, 9}
, {24, -3, -37}
, {57, 5, -65}
}
, {{-91, -59, 28}
, {17, 0, -25}
, {-24, 3, -65}
, {-38, -43, -97}
, {30, -35, -12}
, {-31, -70, -80}
, {-69, -47, 8}
, {16, -44, -51}
, {62, 67, 55}
, {-55, -58, 41}
, {-17, 23, -40}
, {-41, 16, -103}
, {0, -13, -29}
, {-36, 38, -72}
, {-91, -91, -2}
, {-8, -47, 41}
, {-1, 34, -89}
, {-70, -42, -90}
, {-82, -46, -9}
, {-7, -42, -10}
, {-54, -45, 31}
, {-46, 51, -36}
, {-52, -97, -38}
, {-57, -22, -39}
, {-2, -2, -25}
, {-2, -80, -60}
, {-17, 27, -6}
, {-55, -37, -78}
, {-33, -9, -54}
, {-51, -71, -65}
, {-84, 18, 22}
, {-101, -30, -20}
}
, {{-32, -67, -4}
, {-9, -18, -95}
, {-94, -93, 23}
, {49, -28, -68}
, {18, -63, -77}
, {-37, -55, 0}
, {-95, 25, -97}
, {-40, -69, 42}
, {-13, -18, 13}
, {-28, 16, -6}
, {39, 75, -56}
, {-54, -7, 11}
, {10, -101, -87}
, {-15, -93, -86}
, {-41, 49, -44}
, {37, 141, 106}
, {47, -2, -87}
, {13, -47, -32}
, {-104, -51, 17}
, {2, -11, 1}
, {-12, -23, 30}
, {10, 61, 30}
, {-81, 9, 21}
, {-74, -79, 1}
, {-59, 9, -9}
, {-32, -24, -97}
, {27, -13, -41}
, {-23, 1, 38}
, {19, -94, -38}
, {-82, -90, -104}
, {7, -41, -65}
, {-80, -61, 17}
}
, {{-20, -56, -68}
, {-49, -49, -39}
, {-29, 14, -69}
, {9, 36, 41}
, {17, 19, -68}
, {26, 55, 9}
, {-14, -20, -30}
, {0, -30, -48}
, {-62, 17, -55}
, {-17, 9, 47}
, {-136, -144, -87}
, {-85, 19, -6}
, {-8, -57, -58}
, {-1, -31, -9}
, {-44, -57, -5}
, {-67, -56, 3}
, {7, -38, -30}
, {-97, -83, -7}
, {-94, 8, -86}
, {-53, -68, -31}
, {3, -30, 23}
, {-98, -45, 2}
, {-83, -17, -1}
, {-58, -7, -83}
, {46, 44, -13}
, {34, 45, -52}
, {-63, 23, -42}
, {32, -82, 11}
, {-50, 21, -15}
, {54, -41, -23}
, {-31, -133, -85}
, {-1, 36, 18}
}
, {{38, 34, -24}
, {-60, -44, -40}
, {-75, 33, 0}
, {-5, 47, 101}
, {-66, -89, -44}
, {-52, -13, 3}
, {-79, 23, -57}
, {20, 27, -87}
, {-54, -85, 22}
, {13, 8, -19}
, {-2, 41, -8}
, {-89, -100, -33}
, {32, -67, -84}
, {-46, 6, 49}
, {-20, 10, -59}
, {49, 38, 14}
, {37, -64, -83}
, {23, -89, -45}
, {-45, -52, -57}
, {-40, -25, 8}
, {27, -100, 31}
, {-8, -109, -38}
, {-30, -66, 18}
, {-92, 23, -15}
, {-129, -74, -112}
, {-53, -68, -62}
, {-88, 21, 1}
, {-59, -103, -28}
, {-1, -81, -45}
, {-100, -77, 5}
, {-24, 9, 0}
, {-27, -75, -27}
}
, {{-69, -68, 43}
, {36, -64, 14}
, {-81, -90, -43}
, {-42, -91, -58}
, {-100, -3, 33}
, {-62, -24, -83}
, {-19, 31, 36}
, {34, -67, 30}
, {-99, -65, -41}
, {26, 21, -32}
, {-59, 8, -29}
, {27, -7, 31}
, {15, -43, -69}
, {37, 10, -76}
, {-36, -55, 46}
, {-1, -92, -104}
, {-75, 1, 35}
, {-98, -76, 27}
, {3, -48, 15}
, {-71, 30, 45}
, {-50, -76, 12}
, {-90, -41, 4}
, {-26, -97, 44}
, {-99, 40, -34}
, {-96, 3, -97}
, {-60, -5, -76}
, {-97, -82, 29}
, {-59, -5, -8}
, {-69, 21, -24}
, {2, -1, -63}
, {-73, -1, 6}
, {-77, 2, 30}
}
, {{-107, -4, 0}
, {6, -54, -70}
, {-72, -71, -94}
, {15, -97, 53}
, {41, 33, -103}
, {3, 43, 9}
, {26, -61, -56}
, {-33, 34, 37}
, {-4, 17, 57}
, {-16, -38, 27}
, {-77, -67, 72}
, {-1, -45, -16}
, {41, 10, -52}
, {-76, -53, -30}
, {41, 37, -28}
, {-81, -91, 27}
, {-28, -61, -13}
, {-94, -79, 27}
, {-44, -91, 42}
, {-25, -35, 50}
, {-72, 3, -39}
, {-62, -55, 8}
, {-4, -27, -45}
, {-38, -60, 0}
, {136, 70, 96}
, {0, -17, -79}
, {12, -6, 45}
, {69, 20, 82}
, {-13, -67, 0}
, {28, -8, -64}
, {30, -35, 124}
, {-28, -22, -76}
}
, {{-63, -111, -39}
, {-90, -43, 50}
, {24, -64, -88}
, {-118, -19, -11}
, {-36, -113, -36}
, {-76, -73, -3}
, {26, -42, -72}
, {48, -28, -41}
, {68, 13, -60}
, {-82, -70, 9}
, {-37, -25, 49}
, {27, -19, -90}
, {-19, -36, 10}
, {-37, -32, -91}
, {33, -61, -11}
, {24, -74, -10}
, {-62, 49, -87}
, {-63, -91, 67}
, {-7, 65, 7}
, {13, 13, 23}
, {-30, 52, -25}
, {44, 16, 19}
, {-78, -48, 25}
, {-9, 48, 18}
, {-47, -52, -12}
, {42, -15, 11}
, {35, -81, -59}
, {-7, 6, -3}
, {-90, -87, -56}
, {-78, -47, -71}
, {-63, 3, -82}
, {-32, -18, -43}
}
, {{-31, -52, -13}
, {-31, -69, -64}
, {-38, -66, 14}
, {-23, -1, -29}
, {57, 51, 50}
, {-8, -85, 49}
, {40, 2, 40}
, {122, 85, -33}
, {72, 75, 41}
, {-11, 9, 61}
, {-20, -5, 0}
, {64, -2, -33}
, {32, 4, -48}
, {38, 33, -42}
, {157, 47, 122}
, {39, -60, 56}
, {164, 64, 140}
, {-63, 7, 1}
, {-41, -47, 35}
, {-40, -71, 16}
, {-29, -7, -22}
, {-61, -88, -90}
, {65, 26, 72}
, {3, -22, 24}
, {113, -40, 15}
, {-29, -88, -46}
, {12, 50, 59}
, {-3, -39, 60}
, {-6, -32, -50}
, {-17, -40, -94}
, {4, 74, 79}
, {-22, -17, -51}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE