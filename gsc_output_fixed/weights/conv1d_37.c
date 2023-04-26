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


const int16_t conv1d_37_bias[CONV_FILTERS] = {-377, -681, -111, 113, -224, 172, -1053, -48}
;

const int16_t conv1d_37_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-382, -250, -27}
, {-221, -11, -137}
, {-89, 4, -32}
, {40, 74, -110}
, {420, 48, 288}
, {13, 178, 32}
, {-198, -332, 109}
, {7, 77, 103}
}
, {{207, 79, -168}
, {84, 94, -458}
, {472, -141, -387}
, {-257, -99, 163}
, {-149, -103, -273}
, {156, -104, -157}
, {96, -156, -220}
, {170, 28, -48}
}
, {{120, -52, -251}
, {-92, -166, 21}
, {-124, -301, -120}
, {328, -64, -128}
, {-247, -37, 138}
, {-762, -370, -5}
, {-435, -358, 168}
, {-165, -12, -143}
}
, {{-151, 485, 48}
, {-537, -137, -165}
, {29, 136, -223}
, {52, 335, 55}
, {-19, -61, -159}
, {-178, -429, -421}
, {-1009, 43, -317}
, {341, -67, 17}
}
, {{4, -87, 215}
, {-66, 63, 26}
, {-60, -90, 47}
, {-292, -241, -589}
, {-405, -156, 57}
, {-183, -128, 215}
, {136, 98, 323}
, {-506, -396, -184}
}
, {{-231, -119, -569}
, {-221, -36, -329}
, {-381, -193, -464}
, {-584, -393, -71}
, {-283, 166, -323}
, {-44, 184, -194}
, {-100, 329, 33}
, {-295, -246, -364}
}
, {{286, 216, 126}
, {-222, -225, -164}
, {305, 2, -52}
, {101, 38, 54}
, {154, 263, 25}
, {395, 49, -78}
, {-370, -243, -100}
, {186, 42, -229}
}
, {{123, -147, -141}
, {81, 12, 40}
, {-59, -211, -22}
, {103, -122, 212}
, {194, 292, 29}
, {-142, 151, -24}
, {-58, 282, 113}
, {-364, -346, -139}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE