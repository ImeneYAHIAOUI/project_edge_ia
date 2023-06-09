/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_14_bias[CONV_FILTERS] = {6, -84, 180, 59, -180, -135, -67, 175, 54, -82, 37, -33, -78, -184, 2, 304}
;

const int16_t conv1d_14_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-36, -234, 106}
, {-63, -158, -154}
, {67, -2, 96}
, {-62, -168, 43}
, {-129, 101, -18}
, {-101, 76, 67}
, {35, 69, 129}
, {-4, -44, -244}
}
, {{-132, 64, -26}
, {-7, -97, -154}
, {8, -27, 4}
, {173, 113, -83}
, {-157, -30, 113}
, {-200, -98, -152}
, {57, -82, -109}
, {-135, -28, -46}
}
, {{-151, -199, 51}
, {-102, -225, 36}
, {33, 94, -58}
, {-201, 42, 73}
, {-80, -10, 73}
, {-202, 101, 0}
, {-152, 65, 80}
, {-150, -196, 113}
}
, {{1, -224, -84}
, {82, 60, 95}
, {-6, 66, 10}
, {-136, 65, -90}
, {-60, 48, -27}
, {5, 81, 3}
, {-57, 35, 124}
, {-12, 116, -166}
}
, {{44, 38, 1}
, {-110, 126, 85}
, {91, -70, 47}
, {-131, 30, 55}
, {-80, 3, 70}
, {42, -20, 103}
, {-14, 175, 12}
, {-38, 29, 84}
}
, {{55, -87, 75}
, {125, 6, 74}
, {-81, -150, 83}
, {139, -46, -166}
, {149, 93, -123}
, {-211, -296, -32}
, {3, -101, -123}
, {9, 59, 1}
}
, {{26, 83, -49}
, {-212, -78, 7}
, {-126, 88, 56}
, {-89, 124, 97}
, {-228, -22, -95}
, {64, -71, -91}
, {-56, 5, 118}
, {-253, -70, -112}
}
, {{78, 17, 48}
, {-170, -142, -15}
, {-153, -131, 82}
, {-187, -8, -251}
, {177, 157, 59}
, {-29, 10, -60}
, {45, -74, -17}
, {-32, -104, -29}
}
, {{-157, 13, 146}
, {-72, -164, -50}
, {105, 88, -83}
, {-108, -100, -5}
, {-156, -194, 27}
, {70, 120, 94}
, {-60, -15, 39}
, {95, 122, -146}
}
, {{-86, -50, 183}
, {3, -33, -65}
, {52, -59, -22}
, {-96, -101, -213}
, {-18, 95, 151}
, {220, -23, 191}
, {146, -3, 3}
, {104, -99, 73}
}
, {{-45, -222, -260}
, {56, -165, -35}
, {-231, 15, -143}
, {149, -51, 57}
, {207, 112, -112}
, {-35, 94, 125}
, {-48, -2, 98}
, {41, -121, -35}
}
, {{-154, 3, -213}
, {-136, -13, -99}
, {-55, -178, 145}
, {-30, 64, -18}
, {-58, -8, 21}
, {-10, -65, -38}
, {-149, 103, 86}
, {-7, -196, 3}
}
, {{-83, -96, -77}
, {-54, -158, 21}
, {-114, 60, 79}
, {-118, -183, -95}
, {-11, -244, -63}
, {19, -86, 4}
, {-121, 20, 6}
, {-206, -8, 156}
}
, {{-78, -92, 73}
, {49, -34, -69}
, {134, -49, 134}
, {41, 147, 85}
, {-41, 38, 83}
, {-73, 243, 88}
, {39, -106, -23}
, {131, 81, 173}
}
, {{-26, 84, 92}
, {91, -6, -77}
, {-105, -94, -232}
, {-75, 147, -62}
, {175, 116, 112}
, {-55, -232, -245}
, {-79, 139, -219}
, {-102, -45, -236}
}
, {{-14, -25, -68}
, {18, 13, 99}
, {-14, 44, -14}
, {-260, -116, -264}
, {62, 69, -170}
, {14, -146, -134}
, {-192, -185, -61}
, {86, -113, -70}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE