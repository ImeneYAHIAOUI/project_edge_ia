/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      50
#define CONV_KERNEL_SIZE  10


const int16_t conv1d_62_bias[CONV_FILTERS] = {38, -1, -9, 0, 5, 11, -6, -9, -5, 20, 15, -8, -5, 5, 0, 20, 24, 3, -32, 25, 7, 17, 17, 5, 11, 10, 19, 24, 19, 0, -3, -3, 48, -5, 2, -15, -9, 32, 18, -2, -18, 33, -5, -1, 0, -19, -9, 3, 0, 1}
;

const int16_t conv1d_62_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{50, 48, 18, -45, 49, -16, -27, -54, -41, -40}
}
, {{11, 6, 9, -11, -60, -75, -58, -70, -33, -68}
}
, {{-9, 16, 28, 43, 46, 46, 49, -15, 20, 15}
}
, {{50, 38, 46, 32, 26, 56, 52, 37, 26, 3}
}
, {{0, 1, -1, -59, 10, -41, -23, -61, -70, 11}
}
, {{-41, -71, 23, -10, -17, -36, 0, -37, -3, -42}
}
, {{34, 13, 60, -34, 64, -15, -19, 38, 0, 15}
}
, {{-16, 70, 17, 51, 47, -21, -10, 15, 79, 30}
}
, {{-16, -34, -40, -59, -68, 24, 23, -47, -10, -6}
}
, {{-57, 63, -14, -51, 10, -53, -66, -37, -41, -67}
}
, {{-63, 25, 0, 51, -69, 68, -96, 64, -48, 69}
}
, {{-38, 71, 46, -4, -39, 62, 54, 47, 47, -34}
}
, {{-42, 54, 50, -18, 34, 3, 43, 15, 22, 55}
}
, {{63, 7, -20, 53, 33, -23, 33, 21, 12, -45}
}
, {{0, 1, -63, 20, 5, 5, -75, 16, -65, -8}
}
, {{-17, -22, 35, -16, -22, 48, 41, -66, -16, -41}
}
, {{16, -41, -30, 13, -67, 63, -27, 41, -40, 6}
}
, {{11, 63, 46, 38, -30, 9, -17, 30, 50, 67}
}
, {{43, -32, -18, -5, 58, -31, 28, -35, 13, -14}
}
, {{7, 74, 3, 74, 53, 19, 39, 29, 1, 63}
}
, {{42, 56, -46, 27, 29, -16, -48, -88, -63, -22}
}
, {{-33, 13, 8, -1, 0, -70, 29, -67, -47, -27}
}
, {{10, -27, 47, -37, 48, -53, -16, 29, 58, -61}
}
, {{-48, -28, -31, 73, -62, 20, 21, 78, -17, 31}
}
, {{15, -16, -9, 59, 38, 7, -3, -72, -30, -55}
}
, {{7, 63, -69, 53, -47, 29, -51, -30, 53, -26}
}
, {{18, -37, 28, 7, 3, -67, -38, -73, -33, -49}
}
, {{-20, -59, 20, 4, 1, -91, 16, -20, 42, -4}
}
, {{22, 32, 27, 33, -18, 56, -27, 65, 36, 5}
}
, {{-71, -104, -19, 0, -3, -91, 50, -55, 22, -80}
}
, {{-50, 19, -49, -28, -51, 5, 27, -46, 53, -34}
}
, {{-10, 23, -16, 19, -50, 74, -9, 51, 25, 47}
}
, {{49, 40, 36, 46, 60, -9, -21, -31, 43, 35}
}
, {{-65, 19, -43, -28, -17, 23, -20, -53, 59, -62}
}
, {{9, -10, 51, -29, 14, 73, 35, 78, 52, 74}
}
, {{-56, -30, -35, 18, 30, 1, 44, -8, 63, 34}
}
, {{26, -40, 55, -21, 2, 27, 69, 52, 69, -16}
}
, {{-38, -33, 38, -13, 20, -57, -30, 18, 17, -50}
}
, {{1, -25, -18, -39, -32, -5, -66, -25, -27, 34}
}
, {{57, -2, 0, 42, 71, 5, -28, 39, 8, 65}
}
, {{-66, 43, 6, -7, -60, 10, -64, -45, -72, 26}
}
, {{-13, 12, 17, -42, 46, -48, 64, 19, -16, 22}
}
, {{16, -56, 12, -32, 32, -26, 1, -9, -38, -60}
}
, {{-44, 33, -11, -30, -20, -9, -56, 20, -53, 87}
}
, {{18, 12, 69, 52, -25, 21, 23, 0, -33, -89}
}
, {{-14, -55, -66, 15, 1, -14, 26, 51, 89, 36}
}
, {{-6, 19, 20, 79, -8, 72, 12, 64, 33, 50}
}
, {{42, 56, 21, 15, 12, -12, 9, 48, -11, 14}
}
, {{6, -32, 31, 69, -36, 43, -34, -38, -20, -59}
}
, {{-35, 10, -38, 7, -31, -6, 26, -11, -4, -63}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE