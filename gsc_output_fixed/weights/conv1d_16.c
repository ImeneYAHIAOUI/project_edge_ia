/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    10
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  30


const int16_t conv1d_16_bias[CONV_FILTERS] = {44, -59, -50, -86, -30, 33, 23, -46}
;

const int16_t conv1d_16_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{59, -9, -61, 47, 1, 71, -41, 53, 31, 34, -17, -14, -40, 64, -72, 39, 18, -18, -32, 63, 31, 31, 18, 22, -16, 56, 3, 55, 20, 64}
, {85, -20, 92, 23, 50, 41, 53, 42, -9, 11, 46, 32, -22, 17, -16, -40, 44, -47, 51, 8, -51, 48, -20, 27, 52, -16, -51, 60, -61, 10}
, {55, -57, -29, 63, 34, 21, 36, -31, -13, -36, 1, 39, 4, 53, -53, -21, -50, -68, -51, 42, -19, 36, -33, -13, -25, -27, 58, 30, 48, -37}
, {19, 25, 18, 28, 58, -22, 69, 44, -30, 59, 57, 63, -32, -6, 43, -55, 33, 5, 43, -34, -19, 0, -49, 45, 60, 31, 34, 46, -2, -23}
, {20, 3, -1, -28, -40, 79, -36, 61, -12, 26, 25, 3, -24, 32, 44, 13, 33, -54, -51, -28, -33, 70, 32, 23, -35, 31, 32, -23, 58, 49}
, {80, -43, 74, 43, -7, 9, 22, -56, -6, -28, 27, 6, 32, 10, -24, 9, 10, -77, -25, -47, -18, 31, 34, 71, 29, 55, 72, 10, 23, -30}
, {-4, -4, 20, -53, 16, 54, -10, 7, 58, 31, 27, 9, -13, -62, 60, -15, -20, -59, -34, -75, 33, 10, -11, 64, -1, 9, 56, -38, 89, 74}
, {1, 8, 73, 1, 2, -9, 14, 37, 21, 0, 3, 20, 42, 39, -18, 19, -17, -62, -32, -3, -34, -1, 30, -28, 4, 1, 13, 0, 28, -34}
, {-21, -13, 101, -35, 57, 20, 26, -17, 35, 36, -4, 45, 87, -22, 50, -63, 35, -52, -18, -41, 0, 17, 37, 25, 20, -1, 71, 4, 80, 9}
, {46, 76, 14, 31, 62, -63, -6, -29, -40, -19, -6, 45, -6, 46, -15, 19, 46, -17, 17, 21, 29, -17, -8, -41, -17, 35, 1, -16, 38, -7}
}
, {{29, -23, -56, 11, 1, 27, 17, -5, -45, -61, -17, -20, -19, 6, -53, -34, 32, -45, 27, 8, 16, 50, -64, -32, 4, 29, -30, -42, -22, -28}
, {-34, 10, 30, -39, -11, 0, 6, -33, 17, -58, 8, 6, -45, 11, -14, -64, -31, -59, -31, 6, 9, -38, -12, 23, 9, -50, -70, -61, -15, -59}
, {-24, 18, 17, -13, 13, -18, 23, -9, 25, -30, 8, -80, -37, -2, -56, -54, -5, -19, -22, -49, -55, -41, -17, -23, -6, -5, 18, 32, -62, -35}
, {-53, 16, 19, -36, -51, 13, 11, -34, 0, -40, -13, -22, 2, -18, 24, -32, 43, 6, -24, -38, -23, 21, -55, -35, -31, -6, -8, -6, -19, -47}
, {-36, -23, 21, 67, -45, 11, -47, -7, -73, 16, -57, 13, -7, -70, -29, -45, 0, 9, -34, 5, 31, -25, 6, 19, 13, -30, -5, -23, 12, -53}
, {-33, -39, 15, -43, -31, 24, -57, -23, 26, -20, -68, -69, -17, -63, -17, 26, 0, -21, -1, 13, 14, 22, -16, -19, -63, -20, -31, 29, -59, 11}
, {-10, 3, 24, 0, -64, -24, -21, 23, -77, -66, -12, -26, -33, -28, 32, -11, -31, -5, 24, 23, -14, 25, -14, -25, -38, 16, 19, -34, -13, 41}
, {-9, -35, -16, -45, 24, -12, -45, 2, 23, 1, -70, 1, 26, 4, -48, 17, -2, -66, -70, 12, -38, -76, -30, -55, 33, 14, 20, 8, -17, -66}
, {25, 13, 8, -25, -43, -2, -52, -49, 11, 29, 0, 20, -23, -22, -16, -29, 17, -28, 31, -2, 26, -7, 28, -32, -4, 12, -39, -9, -34, 27}
, {-12, -39, -16, 36, -63, -9, 41, -60, -60, -26, 4, 27, -34, -42, -55, -8, 25, 4, 19, 6, 0, 1, 16, -49, -23, -25, 32, -64, 10, 23}
}
, {{56, 2, 6, 24, 70, 49, -38, -28, -75, 19, -12, -21, 31, 0, 24, 10, 43, 53, -49, 66, -85, 21, -32, 35, -3, -21, 29, -35, -37, -15}
, {25, 4, 61, 17, 14, -28, -46, -42, 9, -38, -45, 13, -47, -23, -31, 55, 9, 10, -52, 55, 15, 39, 0, 32, -70, -23, -42, -10, 105, 43}
, {40, 96, 4, 59, 32, 58, -29, 11, -49, 37, 16, -56, 45, 12, 72, 42, 67, -23, 35, 5, -81, -33, -77, 21, -1, -65, -21, -6, -26, 36}
, {41, -11, 5, 9, 67, 0, 41, -41, 16, -37, 32, -16, -13, 20, 29, 52, 29, 39, 32, 44, 7, 43, -40, -52, 17, -28, -15, -8, 90, -28}
, {52, 62, 75, -20, 29, 13, 36, 8, -17, -44, 0, 33, -51, 11, 55, 50, 44, 85, 37, 3, 22, -19, -20, -21, -60, -1, -3, 45, -22, -1}
, {-22, 92, -5, -8, 73, -52, 54, -68, 27, 32, -25, -26, 41, -46, 86, 22, 9, 22, -8, 60, -51, 31, 11, -13, -54, 11, 28, -45, 57, 0}
, {59, 12, -16, -18, -7, -41, -9, -73, 6, -37, 30, -14, 4, 15, 28, 40, 70, 72, -30, -41, 44, -13, -23, -5, -4, 33, -78, 24, 37, 53}
, {2, 50, 70, -13, 40, -15, -19, 10, -3, -46, 31, -47, 41, 29, 75, 97, -5, 67, -10, -25, -1, 12, -20, -22, -8, 0, 32, 29, -10, 40}
, {75, 1, 77, 63, 46, -66, 3, -54, 35, -9, -41, 37, 48, 25, -15, 64, 65, -31, 39, -37, 52, -58, 2, 13, -16, -7, -64, 11, 55, 59}
, {41, -18, 92, 15, 11, -57, 44, -61, -1, -2, -22, 20, 1, -24, 4, 104, 49, 30, 52, 19, 2, -23, 35, -3, -19, 56, -74, 41, 37, -6}
}
, {{-12, -37, 59, 9, -3, 29, 4, 11, -8, 0, -34, -27, -9, -41, 28, 2, 74, -8, 59, 30, 60, 32, 21, -25, 67, -27, -24, -22, -2, -32}
, {-16, 42, 6, 41, 35, 4, 11, -30, -17, -74, -18, -15, -8, -44, 73, -29, 79, 31, 21, -8, -14, 1, 31, -18, -2, 3, 50, 8, -26, -2}
, {0, -16, 58, -13, -15, 35, -25, -42, 25, 0, 40, -18, 33, 0, 10, 10, -2, 11, 82, 12, -11, 21, 30, -8, 4, 11, 15, -40, 61, -26}
, {-25, -14, 4, -27, 2, 30, -18, 19, 39, -57, -31, 1, -19, 1, -17, -1, 30, 54, -3, 20, 69, 0, 8, -20, -15, 33, 46, -59, -1, -16}
, {37, -16, 75, 35, 28, 11, 60, 22, 11, -7, 39, -9, -19, 26, 48, -5, -27, 82, 9, 19, 76, -20, 35, 12, -7, -8, -12, -19, -9, -24}
, {9, 22, 53, 58, 25, 37, 59, -11, -32, 12, 11, 15, 23, 19, 49, 17, -6, 51, 44, 23, 87, -12, 55, -1, 10, 19, -22, -46, 61, 1}
, {52, -16, -12, 5, 36, 36, 41, 26, -38, 5, -17, -29, -43, -16, -17, -11, 53, 82, 35, 77, 55, 24, 40, 12, -19, 11, 2, 7, 28, 36}
, {7, 0, 12, 19, 61, -2, 15, 18, 33, -48, 15, -31, 35, -22, -39, -32, -25, 17, 2, -20, 25, 21, 38, 42, 11, 27, -18, 44, 39, 17}
, {8, -47, -17, 43, 66, -16, -2, -2, -22, 41, 36, -30, -7, -36, 18, 52, 15, 32, 40, -8, -10, 20, 3, 45, -22, -10, 25, -38, 43, 34}
, {27, -21, -15, 28, 34, 29, 57, 32, 28, 40, -43, -31, -24, 20, 30, -13, -58, 67, 14, 13, 37, 25, -44, 28, -7, 27, 34, -20, 22, 4}
}
, {{25, 51, 15, -1, -29, -34, -6, -3, -57, 8, -61, -39, -34, -12, -14, 22, 12, -24, 0, -40, -22, -1, -17, -12, -53, 16, -40, 46, -17, 11}
, {-15, -48, -54, -11, 37, 41, -44, 49, -52, 48, 23, -19, 24, -65, 7, 25, -11, 13, 39, 34, 41, 18, 22, -16, -29, 39, -9, -20, -14, -31}
, {-15, 53, 16, -23, -13, -39, 2, 60, -24, 25, -8, -4, -41, -24, -52, 48, 19, -51, -44, -33, -38, -39, -19, -53, -13, 21, 47, -44, -13, -27}
, {3, 34, -31, -42, 21, -44, -18, -12, -58, 34, -15, -12, 16, 44, -9, 29, 27, 29, 4, -10, 8, 14, -56, 71, -15, 4, -7, -49, -30, 0}
, {-31, -6, 58, 24, -4, -24, 37, 30, 21, -44, -31, -42, -83, 15, 13, -9, 5, -61, 8, 31, 38, -37, 48, -57, -7, -46, 0, 57, 39, -16}
, {2, 9, 13, 0, -39, -35, 21, 50, 40, -44, 65, 4, -17, 37, -1, -26, -35, 28, 15, -12, -23, -14, -16, 22, 70, -30, 31, -47, 15, 51}
, {-51, -37, 62, 19, 25, 33, -34, 6, 36, 5, -10, 4, -30, 8, -28, -2, -58, -22, 39, -4, -24, 22, 50, -13, 63, -8, -29, 54, 17, 39}
, {-93, 41, -25, -63, -52, 2, -66, -40, 57, 19, -33, 16, -48, 63, 30, 32, 1, 47, -33, -17, 31, 17, -41, 7, 0, 16, 0, 58, -36, -1}
, {-31, 16, -37, 30, -15, -24, 46, 9, 54, 5, -29, -11, -54, -12, 21, 25, 12, -13, 43, 28, 45, 13, 28, -6, -37, -24, 39, 43, -22, 52}
, {-54, 30, -5, -3, 13, -48, -54, -70, -11, -31, 18, 26, -8, 5, -16, -45, -9, 38, -8, 12, 26, -33, -14, 25, 54, -46, 9, 4, -51, 4}
}
, {{-28, 4, -27, -54, -37, 27, 60, 38, 13, 47, 21, 6, 20, -28, -22, 46, 9, 70, 34, -24, -20, -45, -16, 0, 42, -32, -28, -3, 17, -21}
, {-45, 8, -3, 43, -33, 36, -51, -48, -69, -71, -31, 16, -31, -82, -38, 6, -58, 32, -69, -20, 14, 37, -47, 8, -8, -86, -82, 47, 47, -29}
, {16, 7, -64, -10, -28, 27, -53, 28, -61, 33, 27, 15, 3, 58, -1, 1, 5, 61, -36, 23, 5, 16, -42, -39, -40, -18, 20, 0, -16, 36}
, {-10, -53, 6, 1, -14, -27, 2, -1, -21, -22, -11, -72, -35, 16, -27, 45, -62, 23, -68, 3, -81, -21, -78, 25, 20, -56, 14, -52, 44, 38}
, {-67, -47, 18, -9, -23, -52, -2, 1, -39, 8, -38, 2, -11, -28, -42, 6, -41, 6, -85, -25, -70, 19, 16, -21, -33, 15, -6, -42, 29, -57}
, {-2, 14, 7, -16, -35, -13, -68, 29, -34, 17, -12, 13, -21, -59, -67, -16, -35, 16, -26, -21, -88, -1, -16, -2, 22, 33, -28, -62, -65, -49}
, {-73, 33, 45, -2, 33, 38, -14, 35, 2, -2, -31, 10, 61, 26, -29, 28, 1, -16, 6, 27, -60, -10, 2, -46, -25, 35, -27, 34, -15, 9}
, {-62, 36, 7, 4, -8, -28, 28, -6, -63, -5, 2, -22, -36, 8, -46, -40, 19, -13, -61, 32, -70, -17, -20, -29, -41, -38, -40, -47, -22, -23}
, {-19, -51, 38, -40, 5, 57, -14, -34, -35, 7, -81, 8, 47, -1, -56, 41, -5, 14, -62, 53, 5, -18, 5, 43, -15, 46, 18, 41, -70, 6}
, {10, -51, -55, 3, 24, -28, -55, 20, -39, -63, -46, 31, -1, 17, -27, -40, 7, -35, 15, 15, 2, 30, -27, -78, -75, -16, 46, 8, -65, -38}
}
, {{0, 2, 4, 27, -8, 57, 41, 41, 46, 108, 4, 63, 58, -52, -28, 4, 30, -1, 25, -1, 7, 12, 24, 32, 35, -12, 15, 51, -8, 3}
, {17, -53, 18, 19, -37, 9, -19, 21, 39, -4, -23, 52, 27, 22, 23, 13, 38, -33, 25, 47, -1, 42, 7, 47, -23, -3, 13, 34, 22, 16}
, {-29, -31, 11, -3, 11, -6, -4, 11, 40, 79, 64, -33, 41, 3, 44, -18, -11, -20, -52, 14, 51, -19, 62, 31, 12, 36, 0, -18, -1, 1}
, {34, -20, 34, -12, -24, -27, 49, 38, -22, 4, 78, -71, 78, -43, 36, 25, 49, -31, 44, 28, -3, -36, -26, -16, 20, 57, -16, 10, -23, 28}
, {31, 44, 7, -47, 44, 16, 5, 53, 16, 77, -26, 1, 5, -47, 0, 31, 29, -2, -5, 43, 62, 32, -10, -39, -45, -55, -34, 11, -25, 26}
, {39, -48, -16, -17, -9, -21, 17, -20, 39, -9, 2, 20, 0, 27, 86, 18, 66, 47, -15, 49, -11, 51, 22, -42, -14, 17, 46, -37, 42, 8}
, {21, 46, 24, -24, -28, 14, -40, 59, 41, -46, 48, 0, -23, -59, 5, -8, 17, -30, 61, 3, 2, 7, 36, 15, 42, -17, -1, -51, -14, 33}
, {23, 29, -23, -1, 22, -28, -20, -18, 44, 12, 18, 20, -6, 10, 63, -74, -28, -19, 40, -5, -21, -24, 63, -36, 3, -15, 7, -38, 0, 39}
, {-32, 14, -26, -52, 16, -46, 2, 7, 12, -27, 25, -13, 54, -32, -27, 68, 2, -9, 57, -54, 6, -12, 18, -38, -10, -34, 60, -34, 23, 1}
, {-6, 8, -44, -38, 24, -41, 22, -12, -29, -18, -25, 25, -25, -49, 23, 11, 57, -45, 29, -41, 61, -19, -11, 37, -8, 23, -16, 32, -15, 25}
}
, {{-49, 5, -17, -24, 61, -50, 10, 5, 1, -9, 5, -34, -73, 42, 60, 21, 36, 38, 7, 3, -1, 21, 28, -6, -68, -48, -51, -10, 64, -59}
, {-39, -37, 12, 8, 28, 19, 53, -76, -17, -35, -41, 0, -15, 0, -2, 73, -53, -12, 3, 2, 10, -4, -35, -98, -3, -74, -17, -58, 2, -58}
, {16, -54, 40, 26, 18, 42, 13, -4, -10, -14, -18, -36, -45, 23, -7, 3, 39, 2, 17, 52, 44, -54, 20, -46, -57, -74, -60, 0, 49, -7}
, {-34, -20, 40, -4, 37, 18, 0, -32, -55, -36, -6, -88, -63, -31, 11, -8, 6, 5, 38, 12, -17, -56, -37, -19, -12, -29, 22, -6, 26, -37}
, {44, 21, -7, 22, 1, 39, -3, 10, -77, 31, 0, 10, -71, -23, -2, 19, 84, 55, 8, 39, -9, 6, -44, -89, 22, -40, -11, -51, 38, 23}
, {34, 6, 24, -33, 10, -50, -53, -31, -46, -53, -31, -12, -11, -28, -37, 12, -34, 7, -38, -29, -43, -5, -4, -50, 17, -81, 13, -78, -6, -16}
, {-51, -36, -29, -63, 35, -21, 25, 34, -33, -69, -14, -58, 22, -37, -61, -5, 9, 42, 11, 48, 44, 16, 0, -74, -11, -49, -11, -65, 87, -51}
, {-53, 14, 34, 19, 30, 44, 2, -64, -1, -22, -69, -75, -13, -33, -19, -7, -2, -10, 11, 10, -12, -43, -50, -26, 9, -42, -1, -29, -41, 45}
, {34, 73, 38, 56, 0, -24, -20, -20, -2, 0, -29, -18, -31, -9, -27, 85, 63, 81, -7, -1, 57, -18, -55, -16, -42, -75, 6, -35, 39, 19}
, {-48, -7, -24, -47, 15, 35, 27, -62, -98, -47, -47, -31, -84, -39, -34, 13, 61, -45, 5, 4, -25, -81, -53, -18, -36, -40, -31, -78, -48, -38}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE