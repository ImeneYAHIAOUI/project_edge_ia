/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    8
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_5_bias[CONV_FILTERS] = {-275, 121, -795, -303, -90, 116, 53, 300}
;

const int16_t conv1d_5_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-120, 34, -63}
, {70, 84, -118}
, {-85, 185, -43}
, {43, 363, 188}
, {-167, 4, -9}
, {-139, 265, 4}
, {-247, 161, 30}
, {-338, -254, -369}
}
, {{92, 140, -17}
, {-51, -580, 399}
, {65, 241, -153}
, {-78, -2, -143}
, {-116, 15, -190}
, {-15, -75, -399}
, {-300, -365, -174}
, {-9, 107, -187}
}
, {{225, -51, -112}
, {132, -35, 70}
, {-41, 53, -155}
, {-28, -141, -255}
, {305, 27, 21}
, {172, 174, 74}
, {-253, -336, -377}
, {300, 165, -175}
}
, {{-82, -240, -127}
, {22, -41, -273}
, {-410, -275, -240}
, {-103, -48, -124}
, {-288, -50, -216}
, {150, 100, -402}
, {356, 202, 80}
, {-91, -31, -146}
}
, {{-284, -101, -170}
, {-94, 92, -99}
, {-118, -253, -46}
, {-13, -158, -34}
, {-100, -215, 47}
, {-126, -17, -28}
, {-165, -60, -162}
, {-155, -48, -111}
}
, {{-172, -186, -304}
, {56, 142, 193}
, {26, 49, 47}
, {-81, -111, 18}
, {70, -84, -54}
, {183, 132, -91}
, {95, 162, 142}
, {-1, -11, -83}
}
, {{317, -102, 149}
, {-73, -280, -126}
, {-81, -555, -331}
, {171, -119, 168}
, {178, -102, -424}
, {-165, -2, 112}
, {120, -382, 217}
, {-351, 7, 64}
}
, {{32, -268, -294}
, {232, 45, -90}
, {-394, -305, -434}
, {-549, -450, -531}
, {111, 17, -59}
, {-471, -237, -279}
, {-117, -257, -837}
, {-291, -48, -44}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE