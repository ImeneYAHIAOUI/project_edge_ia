/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 32
#define FC_UNITS 3


const int16_t dense_3_bias[FC_UNITS] = {7, -18, 32}
;

const int16_t dense_3_kernel[FC_UNITS][INPUT_SAMPLES] = {{27, -192, 3, -82, 123, -11, 26, 154, -318, 58, 135, -106, -102, -141, 265, -88, -188, 103, 133, 201, -38, 96, -20, -139, 5, 65, -21, -140, 0, -103, 96, -78}
, {157, 135, 81, -87, -144, -97, 36, -57, 268, -21, 169, -135, 32, -372, 188, -81, -78, -56, 153, 46, -160, 167, 168, -139, -1, -18, 156, -6, -73, -71, -57, 139}
, {127, 94, -1, 146, 18, 255, 25, 187, -204, 185, -225, -120, -30, 285, -259, -72, -10, -120, 90, 153, 97, 2, -40, 39, -118, -3, 103, -122, 180, 139, -30, 5}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS