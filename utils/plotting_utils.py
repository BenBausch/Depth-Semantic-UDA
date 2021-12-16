# Package imports
import matplotlib
import numpy as np

# names of the cityscapes classes ordered by id, road has id = 0, ..., bicycle has id = 18
# the void class has id = 250 all void (non-valid) cityscapes classes will have this labels
# also classes from other datasets with no equivalent cityscapes classes e.g. synthia's Lanemarking class
cityscapes_class_names = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                              "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                              "truck", "bus", "train", "motorcycle", "bicycle", "ignore_index"]

# used for plotting the semantic classes
# Maps cityscapes ids to specific colors
cityscapes_colors = {0: 'dimgray',
                     1: 'white',
                     2: 'tab:brown',
                     3: 'mediumturquoise',
                     4: 'brown',
                     5: 'slateblue',
                     6: 'red',
                     7: 'green',
                     8: 'lawngreen',
                     9: 'orangered',
                     10: 'lightskyblue',
                     11: 'blue',
                     12: 'navy',
                     13: 'purple',
                     14: 'olive',
                     15: 'yellow',
                     16: 'gold',
                     17: 'firebrick',
                     18: 'lime',
                     250: 'black'}  # last class is the void class

# all cityscapes ids
cityscapes_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 250]

# create the cmap and the boundary norm used for plotting the classes in their specific colors
cityscapes_cmap = matplotlib.colors.ListedColormap([cityscapes_colors[i] for i in cityscapes_class_ids])
cityscapes_cmap_norm = matplotlib.colors.BoundaryNorm(np.arange(len(cityscapes_class_ids) + 1) - 0.5, len(cityscapes_class_ids))