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


const int16_t conv1d_20_bias[CONV_FILTERS] = {-256, -290, -430, -599, -124, -19, -45, -542}
;

const int16_t conv1d_20_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{14, 208, 410, 264, 128, 44, -60, -119, -54, 16, 77, 148, 242, 127, 55, -268, -273, -39, -13, 171, 322, 329, 200, 78, 64, -232, -285, -153, 132, 312}
, {171, -70, 24, 125, 0, -90, -98, -248, -137, 141, 220, 224, 138, 239, 9, -54, -185, -90, -55, -75, -93, 102, 156, 233, 154, -214, -320, -207, -66, 423}
}
, {{198, -3, 460, 287, -174, 189, 240, 90, 550, 389, -272, -229, -155, 84, 651, 269, -404, -72, -55, -154, 259, 351, -120, -122, -228, -339, 30, 150, -171, 221}
, {-77, -72, 287, 324, 99, 107, -285, -447, 22, 122, -4, 61, -167, -332, 46, -2, -35, 167, 195, -216, -39, 72, 209, 435, 146, 70, 121, -13, 25, 418}
}
, {{226, 136, 2, 76, 301, -3, -70, 396, 507, 371, 336, 332, 147, 118, 329, 112, 22, 162, 58, -297, -223, -288, -426, -192, -206, -366, -49, -27, -344, -223}
, {-6, 139, 163, 95, 278, 343, 250, 70, 177, 248, 222, 181, 231, 67, -155, 26, 106, -222, -374, -45, -235, -346, -327, -65, -51, -192, -316, -221, -109, -215}
}
, {{385, 119, -142, -227, 92, 130, 37, 134, -57, -154, -83, 131, 216, 334, 217, -29, -316, -268, -211, -233, -173, 28, 35, -85, 18, 69, 93, 210, 358, 154}
, {638, 227, -5, -137, 46, 96, -43, 27, 65, -167, -66, 123, 68, 124, 66, 70, 45, -95, -88, -131, 3, -4, 36, 110, 180, 250, 414, 365, 189, 87}
}
, {{277, 128, 35, 17, 158, 209, 102, 124, 275, 142, -58, 94, 263, 271, 130, 71, 183, 96, 32, 91, 140, 189, 23, 42, 141, 1, 93, 87, 154, 217}
, {264, 217, 299, 197, 107, 169, 3, 104, 156, 160, 137, 185, 209, 141, 45, 210, 464, 273, 201, 192, 195, 205, 238, 145, 184, 483, 321, 218, 182, 146}
}
, {{207, 61, -187, -100, 103, 120, 29, -147, -114, 85, 86, 110, 94, 234, 174, -22, -113, -178, -55, 266, 409, 298, 73, -273, -344, -109, 228, 429, 263, -140}
, {272, 30, -198, -367, -103, 423, 402, 113, -175, -384, -394, 46, 393, 517, 305, -256, -446, -472, -234, 3, 199, 165, 12, -263, -430, -142, 208, 257, 86, -213}
}
, {{-138, -64, 204, 12, -203, 13, 121, 61, -300, -270, 190, 268, -232, -441, -49, 173, 36, -205, -193, 53, 142, -14, -232, -187, 43, 15, -198, -191, 79, 55}
, {-353, -256, 257, 32, -429, -297, 427, 321, -401, -626, 122, 426, -80, -550, -112, 408, 53, -266, -287, -34, 169, -9, -193, -177, 91, 180, -110, -330, -86, 362}
}
, {{515, 221, -16, 207, 317, 235, 221, 272, 277, 166, 52, -14, -63, 164, 185, 313, 386, 240, 131, 261, 184, 226, 351, 356, 345, 241, 206, 220, 319, 487}
, {254, 27, -185, -319, -306, -141, -30, 206, -22, -78, -54, -256, -284, -68, -165, -90, 112, -203, -455, -226, 2, 59, 126, 81, -121, -34, -270, -346, -144, 54}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE