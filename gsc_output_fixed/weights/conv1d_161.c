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


const int16_t conv1d_161_bias[CONV_FILTERS] = {-171, 159, -37, -159, -170, 47, 95, 97}
;

const int16_t conv1d_161_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{122, -79, -248}
, {40, 114, -110}
, {20, 185, -91}
, {157, -187, 14}
, {-35, 115, 124}
, {4, -136, 65}
, {-171, -203, 219}
, {124, 123, 157}
}
, {{81, -197, -195}
, {28, -218, 97}
, {-326, 54, 234}
, {60, -5, -131}
, {119, -67, -372}
, {-163, -171, 255}
, {-320, -344, -2}
, {-147, -198, 5}
}
, {{12, 58, -9}
, {-277, -93, 2}
, {22, -48, -208}
, {-165, 29, -230}
, {29, 176, -109}
, {195, 117, 75}
, {132, 131, 74}
, {-140, 91, 99}
}
, {{-209, -44, -4}
, {-144, -212, -201}
, {-152, -264, -232}
, {-322, -321, 158}
, {34, -18, -135}
, {-261, -55, 43}
, {-183, -173, -79}
, {-63, -101, -110}
}
, {{-307, -185, 158}
, {145, -8, -92}
, {-50, 126, 140}
, {16, 193, 168}
, {-281, -278, -60}
, {-278, -257, -296}
, {-212, -57, 183}
, {3, 14, 144}
}
, {{101, 0, -7}
, {-24, 264, 106}
, {-150, -55, -271}
, {147, 156, 2}
, {-5, 102, -168}
, {-30, -240, -169}
, {-215, -229, -164}
, {50, -74, -199}
}
, {{-77, -83, 88}
, {-64, 52, 311}
, {-311, 53, -235}
, {-126, 167, 131}
, {-192, -107, -220}
, {-37, 227, 225}
, {-196, 63, -13}
, {161, 50, -42}
}
, {{162, -282, -3}
, {-320, -251, -234}
, {-66, -25, 58}
, {-108, -312, -185}
, {49, 5, -201}
, {180, -105, -151}
, {-57, 76, -63}
, {30, -220, -132}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE