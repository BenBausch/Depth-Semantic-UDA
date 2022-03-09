# --------------------------------------------------------------------------
# In this file, constants, used trough out the whole framework, are defined.
# --------------------------------------------------------------------------
# ignore labels
IGNORE_VALUE_DEPTH = -100
IGNORE_INDEX_SEMANTIC = 250

# class distributions

CITYSCAPES_19_CLASSES_DISTRIBUTION = [0.32639968695760774,  # road 0
                                      0.05386910206129571,  # sidewalk 1
                                      0.2020565193240382,  # building 2
                                      0.005802106136033515,  # wall 3
                                      0.007766301451610918,  # fence 4
                                      0.010945516954950926,  # pole 5
                                      0.0018395637063419118,  # traffic_light 6
                                      0.004880278931946314,  # traffic_sign 7
                                      0.1410130072842125,  # vegetation 8
                                      0.010249921013327206,  # terrain 9
                                      0.03557920792523552,  # sky 10
                                      0.010791173341895352,  # pedestrian 11
                                      0.0011962060367359834,  # rider 12
                                      0.06192123765705013,  # car 13
                                      0.002367729379349396,  # truck 14
                                      0.002082101837927554,  # bus 15
                                      0.002061852687547187,  # train 16
                                      0.0008733976989233193,  # motorcycle 17
                                      0.003664230378735967]  # bicycle 18

CITYSCAPES_16_CLASSES_DISTRIBUTION = [0.32639968695760774,  # road 0
                                      0.05386910206129571,  # sidewalk 1
                                      0.2020565193240382,  # building 2
                                      0.005802106136033515,  # wall 3
                                      0.007766301451610918,  # fence 4
                                      0.010945516954950926,  # pole 5
                                      0.0018395637063419118,  # traffic_light 6
                                      0.004880278931946314,  # traffic_sign 7
                                      0.1410130072842125,  # vegetation 8
                                      0.03557920792523552,  # sky 9
                                      0.010791173341895352,  # pedestrian 10
                                      0.0011962060367359834,  # rider 11
                                      0.06192123765705013,  # car 12
                                      0.002082101837927554,  # bus 13
                                      0.0008733976989233193,  # motorcycle 14
                                      0.003664230378735967]  # bicycle 15

SYNTHIA_19_CLASSES_DISTRIBUTION = [0.18449321239851604,  # road 0
                                   0.19310850112419536,  # sidewalk 1
                                   0.2929401742283749,  # building 2
                                   0.0026926335692539203,  # wall 3
                                   0.0026790478679661295,  # fence 4
                                   0.010384203636793135,  # pole 5
                                   0.0003893244112192055,  # traffic_light 6
                                   0.0010127266981033011,  # traffic_sign 7
                                   0.103087336510533,  # vegetation 8
                                   0.0,  # terrain 9
                                   0.06850084894229426,  # sky 10
                                   0.042455393074608216,  # pedestrian 11
                                   0.0046920743149845776,  # rider 12
                                   0.04058912855193169,  # car 13
                                   0.0,  # truck 14
                                   0.015276061423922167,  # bus 15
                                   0.0,  # train 16
                                   0.0020701608211436107,  # motorcycle 17
                                   0.0021936399863521855]  # bicycle 18

SYNTHIA_16_CLASSES_DISTRIBUTION = [0.18449321239851604,  # road 0
                                   0.19310850112419536,  # sidewalk 1
                                   0.2929401742283749,  # building 2
                                   0.0026926335692539203,  # wall 3
                                   0.0026790478679661295,  # fence 4
                                   0.010384203636793135,  # pole 5
                                   0.0003893244112192055,  # traffic_light 6
                                   0.0010127266981033011,  # traffic_sign 7
                                   0.103087336510533,  # vegetation 8
                                   0.06850084894229426,  # sky 9
                                   0.042455393074608216,  # pedestrian 10
                                   0.0046920743149845776,  # rider 11
                                   0.04058912855193169,  # car 12
                                   0.015276061423922167,  # bus 13
                                   0.0020701608211436107,  # motorcycle 14
                                   0.0021936399863521855]  # bicycle 15

# weights inverse to the frequency of the class in the training dataset

CITYSCAPES_19_CLASSES_WEIGHTS = [3.063728571,  # road 0
                                 18.563516617,  # sidewalk 1
                                 4.949110508,  # building 2
                                 172.351211548,  # wall 3
                                 128.761428833,  # fence 4
                                 91.361610413,  # pole 5
                                 543.607177734,  # traffic_light 6
                                 204.906326294,  # traffic_sign 7
                                 7.091544151,  # vegetation 8
                                 97.561729431,  # terrain 9
                                 28.106302261,  # sky 10
                                 92.668327332,  # person 11
                                 835.976379395,  # rider 12
                                 16.149547577,  # car 13
                                 422.345581055,  # truck 14
                                 480.283935547,  # bus 15
                                 485.000701904,  # train 16
                                 1144.953857422,  # motorcycle 17
                                 272.908599854]  # bicycle 18

CITYSCAPES_16_CLASSES_WEIGHTS = [3.063728571,  # road 0
                                 18.563516617,  # sidewalk 1
                                 4.949110508,  # building 2
                                 172.351211548,  # wall 3
                                 128.761428833,  # fence 4
                                 91.361610413,  # pole 5
                                 543.607177734,  # traffic_light 6
                                 204.906326294,  # traffic_sign 7
                                 7.091544151,  # vegetation 8
                                 28.106302261,  # sky 9
                                 92.668327332,  # person 10
                                 835.976379395,  # rider 11
                                 16.149547577,  # car 12
                                 480.283935547,  # bus 13
                                 1144.953857422,  # motorcycle 14
                                 272.908599854]  # bicycle 15

SYNTHIA_19_CLASSES_WEIGHTS = [5.420253754,  # road 0
                              5.178435802,  # sidewalk 1
                              3.413666248,  # building 2
                              371.383636475,  # wall 3
                              373.266937256,  # fence 4
                              96.300109863,  # pole 5
                              2568.552001953,  # traffic_light 6
                              987.433227539,  # traffic_sign 7
                              9.700512886,  # vegetation 8
                              0.000000000,  # terrain 9
                              14.598360062,  # sky 10
                              23.554132462,  # pedestrian 11
                              213.125350952,  # rider 12
                              24.637140274,  # car 13
                              0.000000000,  # truck 14
                              65.461898804,  # bus 15
                              0.000000000,  # train 16
                              483.054260254,  # motorcycle 17
                              455.863311768]  # bicycle 18

SYNTHIA_16_CLASSES_WEIGHTS = [5.420253754,  # road 0
                              5.178435802,  # sidewalk 1
                              3.413666248,  # building 2
                              371.383636475,  # wall 3
                              373.266937256,  # fence 4
                              96.300109863,  # pole 5
                              2568.552001953,  # traffic_light 6
                              987.433227539,  # traffic_sign 7
                              9.700512886,  # vegetation 8
                              14.598360062,  # sky 9
                              23.554132462,  # pedestrian 10
                              213.125350952,  # rider 11
                              24.637140274,  # car 12
                              65.461898804,  # bus 13
                              483.054260254,  # motorcycle 14
                              455.863311768]  # bicycle 15
