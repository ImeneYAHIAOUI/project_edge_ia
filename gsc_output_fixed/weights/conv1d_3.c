/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_3_bias[CONV_FILTERS] = {-1289, -495, 6, -276, -5, -437, -609, -411, -1262, 179, 947, 1255, -390, -726, -216, -47, -22, -800, -269, -436, -283, -165, 1376, -760, -262, -768, -103, -740, -617, -339, -706, -579}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{188, -95, -54}
, {92, -17, 33}
, {-199, 10, 41}
, {12, 52, -30}
, {46, -15, -109}
, {-470, -411, 20}
, {29, -111, -104}
, {462, 177, 131}
, {-203, -11, 270}
, {-145, -199, 22}
, {-69, -12, -40}
, {-276, 379, 423}
, {-33, -74, -58}
, {-492, -115, -105}
, {8, 90, -155}
, {-240, -149, -91}
}
, {{80, 42, -92}
, {67, 27, -64}
, {134, -19, 100}
, {301, 88, -429}
, {54, -41, -27}
, {-398, -272, -565}
, {-130, -36, -281}
, {109, 230, 231}
, {-211, 117, -253}
, {154, -152, -84}
, {-109, -215, -165}
, {2, -85, -105}
, {-79, 66, -134}
, {-397, -269, -382}
, {77, -21, 156}
, {-174, -288, -320}
}
, {{18, -111, -146}
, {62, 81, 98}
, {-45, -115, -151}
, {82, -115, -146}
, {16, -24, -19}
, {-870, 568, 436}
, {-161, -37, -79}
, {-22, 217, 477}
, {322, -1504, 270}
, {-367, -342, -220}
, {-191, -314, -445}
, {-330, -715, 584}
, {-1, -143, -119}
, {-508, -479, -206}
, {-22, -25, -83}
, {-1165, -69, -392}
}
, {{-72, -153, -5}
, {-68, -36, -19}
, {-23, -6, 10}
, {-67, -243, -172}
, {-33, -8, 109}
, {-90, -5, -157}
, {70, -64, -129}
, {-150, 11, -20}
, {-168, -96, -154}
, {-56, -2, -152}
, {-166, -74, -343}
, {-219, 18, -142}
, {-43, -38, -4}
, {-28, -183, -181}
, {27, 57, -142}
, {-307, -112, -142}
}
, {{-36, 282, -63}
, {-45, -5, 23}
, {101, 503, 337}
, {14, -130, -143}
, {33, 30, -144}
, {233, 1273, -3081}
, {-105, 222, 209}
, {251, 252, -596}
, {-1403, -2720, -1058}
, {-155, -123, 89}
, {-216, -228, -1664}
, {256, -41, -1548}
, {-56, 183, 119}
, {-500, -102, -97}
, {55, -135, -117}
, {-112, 12, -53}
}
, {{112, -64, 392}
, {-48, -146, -112}
, {-113, -147, -40}
, {-73, -102, -15}
, {-45, -4, -7}
, {377, -489, 166}
, {-23, -129, -55}
, {247, -87, -130}
, {428, -638, -404}
, {-96, -61, 21}
, {-417, -156, -111}
, {388, 136, 52}
, {-99, -45, 44}
, {-297, -120, -98}
, {-79, 21, -4}
, {-204, -713, -167}
}
, {{-94, -204, 54}
, {-11, -104, -60}
, {-33, 226, 112}
, {266, 217, -53}
, {-65, 87, 24}
, {-145, -900, 408}
, {-170, -229, 6}
, {248, -158, 83}
, {82, -298, 645}
, {-9, 14, -60}
, {-322, 0, -246}
, {-764, -43, 241}
, {-21, 137, 26}
, {-235, -120, -211}
, {-152, -34, -176}
, {-393, -881, -343}
}
, {{137, 296, 155}
, {90, -61, -60}
, {-61, -112, -184}
, {-226, 12, 45}
, {10, -74, 196}
, {-86, -113, -153}
, {100, 177, -220}
, {35, 258, -392}
, {-387, -50, -304}
, {-83, -166, -25}
, {-84, -418, -351}
, {-382, -232, -151}
, {-12, -155, -7}
, {-117, -360, -279}
, {-68, 117, -73}
, {-29, -324, -289}
}
, {{-38, 27, 40}
, {-66, 60, -54}
, {-369, -488, -47}
, {-12, -70, -89}
, {62, -101, -10}
, {71, 269, 376}
, {-13, 46, 294}
, {9, 334, -507}
, {101, 271, -771}
, {-464, -172, -183}
, {-822, -678, -778}
, {-163, 468, -327}
, {37, -117, -105}
, {-201, -290, -189}
, {-26, 38, 16}
, {-18, -229, -628}
}
, {{-1, 317, 105}
, {178, -124, -65}
, {192, -192, -157}
, {-182, -207, -9}
, {-109, 100, 173}
, {-392, 157, 54}
, {43, 4, 422}
, {-339, 12, -604}
, {75, -688, -722}
, {138, -311, -475}
, {-539, -289, -564}
, {370, 570, -1053}
, {60, 75, 43}
, {-710, -449, -595}
, {-84, 21, 161}
, {-349, -673, -425}
}
, {{2, -152, -189}
, {-107, -89, -4}
, {-57, -59, -59}
, {-119, -104, -7}
, {-290, -300, -131}
, {780, 608, -1585}
, {-41, -190, 63}
, {429, 191, -257}
, {-1063, -1050, 347}
, {-29, 133, -242}
, {147, -41, 71}
, {-777, -263, -53}
, {-46, -75, 80}
, {-72, -21, -468}
, {29, 41, 0}
, {-50, -141, 2}
}
, {{-323, -197, -194}
, {51, 82, -23}
, {-27, 26, 37}
, {77, 25, 43}
, {89, -219, 30}
, {-369, -1494, -542}
, {227, -127, 39}
, {393, -1114, -554}
, {-504, -725, -288}
, {93, -174, -5}
, {-61, 20, -168}
, {-192, 29, 86}
, {92, -178, 23}
, {-75, -9, -89}
, {-21, -65, 104}
, {48, 14, 144}
}
, {{201, 35, -127}
, {2, -150, 20}
, {-117, -153, -145}
, {-103, -50, 129}
, {16, 30, -89}
, {31, -345, -40}
, {-66, -92, -1}
, {170, -268, -58}
, {-175, 226, 403}
, {29, -144, -270}
, {-164, -265, -738}
, {-419, 107, -335}
, {-78, -77, 9}
, {-502, -233, -306}
, {120, -1, -94}
, {-301, -135, -259}
}
, {{-83, -7, 0}
, {-76, 102, -103}
, {-20, -23, -317}
, {-152, 60, 18}
, {-135, 49, -92}
, {-446, -338, -356}
, {-126, 18, -122}
, {8, 21, -9}
, {-282, -118, -285}
, {-87, -59, -333}
, {-477, -127, -449}
, {36, -109, -404}
, {-31, -149, -30}
, {-299, -217, -156}
, {-100, 84, -15}
, {-242, -132, -143}
}
, {{139, 103, 110}
, {28, -104, 53}
, {202, -90, -256}
, {307, -48, -13}
, {-98, 111, 210}
, {-150, -289, -217}
, {-36, -14, -158}
, {106, 66, -60}
, {-317, -76, -89}
, {54, -82, 262}
, {-198, -209, -229}
, {-177, -141, 3}
, {145, 256, 231}
, {-671, -301, -76}
, {49, -86, -85}
, {-69, -106, -233}
}
, {{32, -37, -73}
, {-47, -79, -54}
, {-110, -79, -111}
, {-66, 16, -85}
, {31, -137, -12}
, {-121, -141, -33}
, {92, -67, -66}
, {-93, -156, -123}
, {-148, -82, 11}
, {-37, -142, -95}
, {-4, -64, -88}
, {-7, -88, -75}
, {-30, 2, 38}
, {-48, -19, -51}
, {-92, -33, -156}
, {-92, -127, -100}
}
, {{-41, -144, 73}
, {7, 83, 32}
, {-33, 54, -102}
, {-110, -6, -146}
, {-204, -165, 32}
, {-263, 4, -550}
, {159, -60, 128}
, {-45, -583, -60}
, {-311, -518, -219}
, {-228, -428, 129}
, {-526, -280, -44}
, {-437, -443, -10}
, {35, 70, 283}
, {-234, -714, -465}
, {-86, -19, 122}
, {-207, -98, -155}
}
, {{-152, 12, -30}
, {15, -39, -6}
, {65, 218, -112}
, {4, -73, 146}
, {-89, 35, 18}
, {-326, -313, -220}
, {133, -133, 166}
, {281, -3, 330}
, {-341, -7, -199}
, {-95, -67, -208}
, {-492, -790, -244}
, {-310, 359, -141}
, {51, 63, 73}
, {-310, -630, -266}
, {-69, -123, -137}
, {-510, -719, -116}
}
, {{-107, -53, 66}
, {16, -5, -6}
, {-12, -69, -24}
, {26, 29, 12}
, {-18, -9, -68}
, {-250, -134, 24}
, {-37, -104, 19}
, {-26, -75, -107}
, {56, -88, -136}
, {-10, -47, 14}
, {-97, -52, -122}
, {-37, -133, -116}
, {-15, -10, 58}
, {-296, -313, -129}
, {-64, -117, -7}
, {-360, -231, -38}
}
, {{-38, -127, 87}
, {-7, 51, -20}
, {-31, -135, 43}
, {49, 26, -214}
, {-6, -31, 283}
, {-213, 4, -247}
, {113, -5, -46}
, {-124, -94, 99}
, {-110, 13, -128}
, {-108, -132, -6}
, {-114, -188, -236}
, {-3, -17, -4}
, {5, -208, 66}
, {-52, -156, -155}
, {51, 31, -48}
, {-416, -349, -65}
}
, {{-41, 29, 62}
, {64, 73, -131}
, {-52, -245, -192}
, {-146, -75, -25}
, {-15, -112, -74}
, {-182, -117, -39}
, {-78, -15, -238}
, {-117, -85, -37}
, {-60, -8, 21}
, {-154, -154, 2}
, {-250, -49, -5}
, {-41, -25, -152}
, {-17, 234, -82}
, {-128, -165, -246}
, {-115, -5, -16}
, {-344, -172, -104}
}
, {{-47, -94, 139}
, {-39, 61, 0}
, {-270, -84, -94}
, {227, -151, -27}
, {-51, -94, -208}
, {521, 62, 129}
, {-139, 122, 86}
, {133, -71, -92}
, {-89, -125, -16}
, {-556, -131, -308}
, {-224, -257, -1315}
, {-429, 338, 150}
, {148, 93, 355}
, {-901, -146, -180}
, {-158, 21, -84}
, {-279, -24, 49}
}
, {{111, 312, 209}
, {6, -11, -197}
, {158, -257, -86}
, {-127, -171, -106}
, {92, -133, 206}
, {-661, 902, -495}
, {-285, -175, 32}
, {-375, -156, 252}
, {171, -524, -402}
, {217, -286, -174}
, {-17, -41, 43}
, {293, -179, -286}
, {58, -151, -84}
, {-241, -53, -385}
, {97, -100, -41}
, {-62, 64, -180}
}
, {{-92, 268, 47}
, {12, -49, 122}
, {80, -24, -79}
, {-141, 434, -99}
, {-285, -182, -92}
, {-330, -299, -716}
, {153, -113, 48}
, {48, -140, -87}
, {161, -715, 470}
, {-31, -337, -148}
, {-106, -659, -277}
, {175, -521, -473}
, {60, 74, 151}
, {-157, -527, -299}
, {-9, 23, 248}
, {-209, -755, -205}
}
, {{185, -110, -16}
, {47, 18, -58}
, {28, -58, -237}
, {-145, 59, -93}
, {-73, 5, -45}
, {-18, 12, -66}
, {-112, -36, -71}
, {-106, -151, 324}
, {24, -90, -69}
, {-174, -414, -61}
, {-195, -284, -143}
, {-153, -167, -26}
, {-25, -132, 195}
, {-92, -171, -310}
, {-107, -148, -122}
, {-1, -261, -133}
}
, {{56, -113, -86}
, {-106, -13, 9}
, {-16, 280, -72}
, {-80, 28, -46}
, {30, 64, -12}
, {-274, -204, -45}
, {-103, -57, -7}
, {64, -121, -125}
, {-22, -229, 278}
, {-80, -53, -134}
, {-91, -183, -196}
, {-415, -98, -163}
, {239, 95, -147}
, {-339, -194, -367}
, {22, -51, 67}
, {-104, -189, -348}
}
, {{-52, -71, -37}
, {-105, -108, -157}
, {30, 3, -23}
, {-85, 149, -240}
, {-164, 22, -227}
, {-92, -190, -268}
, {-151, -84, -85}
, {-233, -19, -139}
, {-117, -127, -396}
, {3, -84, -33}
, {-93, -240, -52}
, {-14, -119, -195}
, {-152, -44, 5}
, {-339, -23, -113}
, {-118, -40, 23}
, {-242, -162, -110}
}
, {{13, -43, -209}
, {-2, -33, -42}
, {-1, -101, -59}
, {-22, -140, -246}
, {-114, 105, -92}
, {-910, -869, 186}
, {-49, -110, 128}
, {-31, -409, 229}
, {-229, -132, -270}
, {-4, -390, -100}
, {-157, -192, -476}
, {-56, -173, -532}
, {-263, -87, -39}
, {-215, -427, -770}
, {-58, -58, -16}
, {-319, -127, -134}
}
, {{-95, -92, 70}
, {64, -51, -102}
, {-316, 272, -81}
, {-31, 221, 225}
, {39, -102, -57}
, {-567, -90, -22}
, {-35, -51, 121}
, {92, -167, -156}
, {-91, -57, -121}
, {-194, -108, -81}
, {-238, -129, -342}
, {-40, -75, -109}
, {-37, -172, -19}
, {-380, -238, -230}
, {-51, 19, 42}
, {-338, -82, -184}
}
, {{-146, 9, -29}
, {-55, 26, -72}
, {-132, -3, -64}
, {-103, -103, 15}
, {-41, 24, 36}
, {-592, -38, -322}
, {-148, -114, -53}
, {-8, -108, 33}
, {-218, -130, -44}
, {-77, 46, 0}
, {-290, -103, -102}
, {-32, 8, -44}
, {-92, 61, 54}
, {-334, -442, -128}
, {-3, 47, -101}
, {-123, -7, 0}
}
, {{71, 27, -1}
, {-44, -109, 38}
, {-358, -555, -198}
, {-82, 76, -136}
, {-58, 53, -31}
, {-214, 146, -1023}
, {61, 129, -37}
, {279, -129, -162}
, {136, -248, -506}
, {-142, -197, -249}
, {-53, -288, -158}
, {-139, -313, -296}
, {66, 34, -53}
, {-649, -390, -198}
, {-40, -178, 146}
, {-39, -250, -512}
}
, {{113, 51, -23}
, {-71, 84, 68}
, {45, -6, 72}
, {-49, 35, -81}
, {-62, 49, -46}
, {-412, 50, -418}
, {-184, -35, -7}
, {109, -121, 60}
, {-299, -189, 89}
, {-128, -104, 14}
, {-394, -416, -582}
, {-286, -15, -126}
, {23, -94, 23}
, {-144, -486, -131}
, {17, 34, 65}
, {-37, -240, -467}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE