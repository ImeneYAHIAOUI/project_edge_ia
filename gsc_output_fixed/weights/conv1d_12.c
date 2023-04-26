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


const int16_t conv1d_12_bias[CONV_FILTERS] = {-3, -20, 26, -12, -3, 8, -94, -2}
;

const int16_t conv1d_12_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-1668, -923, -458, -130, 141, 904, 1102, 819, 308, -66, -876, -1364, -1343, -1379, -1020, -228, 454, 763, 717, 529, 359, 115, -497, -815, -878, -908, -727, -297, 236, 718}
, {-1661, -876, -176, 486, 929, 991, 937, 583, 279, 180, 41, -397, -582, -535, -453, 84, 780, 944, 896, 682, 63, -484, -838, -811, -669, -417, -363, -199, 348, 1062}
}
, {{322, -301, -680, -908, -822, -489, 117, 577, 781, 717, 197, -268, -367, -454, -467, -118, 617, 1013, 1372, 1341, 708, -158, -573, -548, 38, 1221, 1517, 1436, 818, -75}
, {-240, -716, -900, -768, -622, 91, 478, 603, 698, 655, 572, 518, 296, 293, 593, 993, 1115, 1176, 1177, 1083, 674, 313, 32, 1, 401, 991, 1264, 1331, 1204, 525}
}
, {{865, 586, 951, 1221, 1398, 1386, 1338, 1401, 1454, 1414, 1297, 1222, 994, 744, 629, 612, 870, 1070, 1124, 1217, 1261, 1453, 1518, 1351, 1147, 912, 636, 298, 102, 60}
, {963, 375, 468, 754, 944, 1008, 1133, 1393, 1404, 1447, 1491, 1507, 1564, 1374, 1234, 1138, 1200, 1222, 1171, 1271, 1407, 1438, 1444, 1535, 1520, 1283, 1069, 910, 1045, 1374}
}
, {{-429, -1565, -1840, -1189, 110, 1450, 1568, 1362, 895, 204, -597, -893, -905, -1090, -1008, -151, 545, 675, 713, 763, 602, 61, -297, -521, -681, -401, 669, 1401, 1607, 1374}
, {-609, -237, 196, 601, 920, 651, 363, 147, -238, -448, -113, 101, 249, 507, 674, 423, 235, 94, -106, 133, 162, -350, -604, -652, -370, 345, 618, 541, 577, 338}
}
, {{-1224, -947, -450, -22, 791, 1708, 1714, 1465, 1135, 756, 3, -689, -830, -866, -904, -501, 210, 470, 569, 716, 842, 727, 330, 86, -127, -234, 124, 599, 902, 1473}
, {-1172, -591, -145, 233, 618, 665, 647, 753, 666, 708, 252, -442, -981, -1215, -1344, -1073, -600, -103, 195, 327, 377, 147, -242, -347, -334, -375, -430, -624, -411, -78}
}
, {{-938, -277, 507, 676, 277, -499, -1211, -1465, -1694, -1710, -1280, -722, -309, -291, -586, -1082, -1306, -1195, -914, -434, 343, 597, 659, 653, 511, 357, 287, 362, 693, 1266}
, {-381, -241, 182, 486, 278, -286, -1075, -1457, -1528, -1454, -1118, -597, -124, -344, -867, -1337, -1227, -954, -299, 522, 759, 611, 470, 107, -9, 164, 541, 806, 1210, 1662}
}
, {{205, -162, -1087, -1542, -1288, -938, -345, 435, 1012, 1276, 1225, 1092, 351, -739, -1066, -1143, -959, -504, 271, 805, 981, 983, 532, -303, -877, -1126, -1504, -1770, -1199, -205}
, {1385, 666, -282, -622, -832, -1106, -928, -283, 85, 317, 420, 42, -210, -517, -910, -991, -878, -768, -370, 384, 656, 914, 998, 575, -383, -1388, -2247, -2426, -1890, -946}
}
, {{160, 542, 1022, 1621, 1727, 1419, 921, 589, 493, 367, 125, 297, 615, 750, 954, 887, 369, -525, -975, -1039, -1005, -820, -818, -644, -466, -443, -583, -860, -1327, -1564}
, {892, 1456, 1796, 1961, 1899, 1357, 831, 591, 355, 69, 36, 254, 464, 632, 842, 837, 385, -446, -854, -964, -1072, -898, -756, -383, -188, -128, -445, -933, -1577, -1961}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE