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


const int16_t conv1d_209_bias[CONV_FILTERS] = {-154, -59, 50, -86, 75, 53, 94, -30, -56, 49, 161, -54, -92, -253, -66, 21}
;

const int16_t conv1d_209_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-82, -246, -199}
, {48, -112, 78}
, {93, -80, -174}
, {181, -94, -228}
, {-42, 94, 2}
, {131, 88, 83}
, {146, -135, 69}
, {-89, -38, 54}
}
, {{20, 46, -83}
, {-190, -114, -90}
, {-10, -154, -5}
, {-159, -147, -44}
, {-212, 28, 6}
, {-68, -1, -214}
, {-144, -51, -38}
, {-40, -89, -148}
}
, {{-34, 12, 59}
, {-392, -269, -111}
, {73, 154, -225}
, {188, 73, 78}
, {-224, -179, -356}
, {-157, -33, -34}
, {-258, -88, 64}
, {-133, -68, 31}
}
, {{-36, -172, -145}
, {-28, -30, 55}
, {-114, -97, -95}
, {-207, -39, -136}
, {-167, 53, -102}
, {-153, -170, -202}
, {76, -131, 84}
, {-44, -9, 106}
}
, {{-57, 84, -15}
, {-174, -258, -124}
, {-258, -270, -355}
, {-41, -142, 18}
, {-116, 55, 45}
, {-114, -107, 98}
, {-32, -61, -132}
, {-22, 77, -14}
}
, {{-326, -172, -61}
, {-111, 130, 82}
, {84, -57, 229}
, {-305, -107, -171}
, {-44, -245, -95}
, {-75, 116, -75}
, {0, -12, 108}
, {-10, -95, 56}
}
, {{-124, -125, 12}
, {-79, -238, 32}
, {97, -70, 84}
, {75, 88, -122}
, {138, 123, -36}
, {-144, -173, -79}
, {-140, -185, 137}
, {-63, 35, -192}
}
, {{131, 148, -217}
, {167, 164, -4}
, {23, -135, -170}
, {-12, 67, 148}
, {66, -150, 18}
, {-96, -16, 16}
, {61, 100, 65}
, {-142, 106, 134}
}
, {{-124, -125, 77}
, {-211, -160, -45}
, {-164, -17, -177}
, {-273, -201, 5}
, {179, 105, -117}
, {-29, -94, -12}
, {-71, 125, -80}
, {-86, 56, 142}
}
, {{-75, 90, -61}
, {-145, -62, 76}
, {-29, -187, 172}
, {-37, -113, -6}
, {-18, 91, 179}
, {-139, -125, 150}
, {-102, 34, -93}
, {136, 45, -173}
}
, {{165, -41, 116}
, {144, -57, -155}
, {-255, -207, -312}
, {-125, -142, 24}
, {126, 18, -120}
, {-72, -24, -269}
, {-42, -118, -163}
, {-24, -26, -149}
}
, {{-21, -29, 15}
, {-47, 13, -219}
, {-9, 120, -155}
, {100, -15, 123}
, {105, -126, 91}
, {-100, -29, -98}
, {-19, 109, -12}
, {-139, 104, -110}
}
, {{-112, 9, -49}
, {-197, -162, -97}
, {-143, -21, -34}
, {-187, -154, -20}
, {93, -32, 98}
, {76, -120, -164}
, {-64, 158, -135}
, {1, 106, -126}
}
, {{33, -211, 153}
, {161, -27, 51}
, {143, -1, -88}
, {-54, -28, -157}
, {-130, 175, 68}
, {-14, -208, -217}
, {18, 1, 83}
, {-139, 47, -14}
}
, {{-37, -114, 67}
, {24, -50, 11}
, {-123, 62, -177}
, {-128, -80, 17}
, {-95, -6, 3}
, {-136, 23, -77}
, {107, 77, -147}
, {6, -84, 38}
}
, {{30, 34, 67}
, {132, -41, -144}
, {-21, -28, -233}
, {98, -78, 15}
, {166, -123, -171}
, {135, -133, -189}
, {5, 107, -46}
, {193, -83, -65}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE