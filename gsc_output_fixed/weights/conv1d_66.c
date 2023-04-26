/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      5
#define CONV_KERNEL_SIZE  5


const int16_t conv1d_66_bias[CONV_FILTERS] = {-45, -52, -46, -37, -19}
;

const int16_t conv1d_66_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-106, -276, 55, 139, 160}
}
, {{-9, 110, -152, 267, -104}
}
, {{-131, 105, -13, 115, -112}
}
, {{91, 194, 124, 289, -80}
}
, {{138, 23, -197, 92, 0}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE