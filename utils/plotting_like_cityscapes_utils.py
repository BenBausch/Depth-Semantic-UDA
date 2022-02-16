# Package imports
import matplotlib
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# names of the cityscapes classes ordered by id, road has id = 0, ..., bicycle has id = 18
# the void class has id = 250 all void (non-valid) cityscapes classes will have this labels
# also classes from other datasets with no equivalent cityscapes classes e.g. synthia's Lanemarking class
import torch

# ********************************************************************
# /----- Cityscapes with 19 classes
# ********************************************************************

CITYSCAPES_CLASS_NAMES_19 = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                             "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                             "truck", "bus", "train", "motorcycle", "bicycle", "ignore_index"]

# used for plotting the semantic classes
# Maps cityscapes ids to specific colors, ids to names and colors to names
CITYSCAPES_ID_TO_COLOR_19 = {0: np.asarray([128, 64, 128]),
                             1: np.asarray([244, 35, 232]),
                             2: np.asarray([70, 70, 70]),
                             3: np.asarray([102, 102, 156]),
                             4: np.asarray([190, 153, 153]),
                             5: np.asarray([153, 153, 153]),
                             6: np.asarray([250, 170, 30]),
                             7: np.asarray([220, 220, 0]),
                             8: np.asarray([107, 142, 35]),
                             9: np.asarray([152, 251, 152]),
                             10: np.asarray([70, 130, 180]),
                             11: np.asarray([220, 20, 60]),
                             12: np.asarray([255, 0, 0]),
                             13: np.asarray([0, 0, 142]),
                             14: np.asarray([0, 0, 70]),
                             15: np.asarray([0, 60, 100]),
                             16: np.asarray([0, 80, 100]),
                             17: np.asarray([0, 0, 230]),
                             18: np.asarray([119, 11, 32]),
                             250: np.asarray([0, 0, 0])}  # last class is the void class

CITYSCAPES_ID_TO_NAME_19 = {0: "road",
                            1: "sidewalk",
                            2: "building",
                            3: "wall",
                            4: "fence",
                            5: "pole",
                            6: "traffic_light",
                            7: "traffic_sign",
                            8: "vegetation",
                            9: "terrain",
                            10: "sky",
                            11: "person",
                            12: "rider",
                            13: "car",
                            14: "truck",
                            15: "bus",
                            16: "train",
                            17: "motorcycle",
                            18: "bicycle",
                            250: "ignore_index"}  # last class is the void class (Dont change order of insert!)

CITYSCAPES_COLOR_TO_NAME_19 = {(128, 64, 128): "road",
                               (244, 35, 232): "sidewalk",
                               (70, 70, 70): "building",
                               (102, 102, 156): "wall",
                               (190, 153, 153): "fence",
                               (153, 153, 153): "pole",
                               (250, 170, 30): "traffic_light",
                               (220, 220, 0): "traffic_sign",
                               (107, 142, 35): "vegetation",
                               (152, 251, 152): "terrain",
                               (70, 130, 180): "sky",
                               (220, 20, 60): "person",
                               (255, 0, 0): "rider",
                               (0, 0, 142): "car",
                               (0, 0, 70): "truck",
                               (0, 60, 100): "bus",
                               (0, 80, 100): "train",
                               (0, 0, 230): "motorcycle",
                               (119, 11, 32): "bicycle",
                               (0, 0, 0): "ignore_index"}

# all cityscapes ids
CITYSCAPES_TRAINING_IDS_19 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 250]

# ********************************************************************
# /----- Cityscapes with 16 classes
# ********************************************************************

CITYSCAPES_CLASS_NAMES_16 = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                             "traffic_sign", "vegetation", "sky", "person", "rider", "car", "bus", "motorcycle",
                             "bicycle", "ignore_index"]

# used for plotting the semantic classes
# Maps cityscapes ids to specific colors, ids to names and colors to names
CITYSCAPES_ID_TO_COLOR_16 = {0: np.asarray([128, 64, 128]),
                             1: np.asarray([244, 35, 232]),
                             2: np.asarray([70, 70, 70]),
                             3: np.asarray([102, 102, 156]),
                             4: np.asarray([190, 153, 153]),
                             5: np.asarray([153, 153, 153]),
                             6: np.asarray([250, 170, 30]),
                             7: np.asarray([220, 220, 0]),
                             8: np.asarray([107, 142, 35]),
                             9: np.asarray([70, 130, 180]),
                             10: np.asarray([220, 20, 60]),
                             11: np.asarray([255, 0, 0]),
                             12: np.asarray([0, 0, 142]),
                             13: np.asarray([0, 60, 100]),
                             14: np.asarray([0, 0, 230]),
                             15: np.asarray([119, 11, 32]),
                             250: np.asarray([0, 0, 0])}  # last class is the void class (Dont change order of insert!)

CITYSCAPES_ID_TO_NAME_16 = {0: "road",
                            1: "sidewalk",
                            2: "building",
                            3: "wall",
                            4: "fence",
                            5: "pole",
                            6: "traffic_light",
                            7: "traffic_sign",
                            8: "vegetation",
                            9: "sky",
                            10: "person",
                            11: "rider",
                            12: "car",
                            13: "bus",
                            14: "motorcycle",
                            15: "bicycle",
                            250: "ignore_index"}  # last class is the void class

CITYSCAPES_COLOR_TO_NAME_16 = {(128, 64, 128): "road",
                               (244, 35, 232): "sidewalk",
                               (70, 70, 70): "building",
                               (102, 102, 156): "wall",
                               (190, 153, 153): "fence",
                               (153, 153, 153): "pole",
                               (250, 170, 30): "traffic_light",
                               (220, 220, 0): "traffic_sign",
                               (107, 142, 35): "vegetation",
                               (70, 130, 180): "sky",
                               (220, 20, 60): "person",
                               (255, 0, 0): "rider",
                               (0, 0, 142): "car",
                               (0, 60, 100): "bus",
                               (0, 0, 230): "motorcycle",
                               (119, 11, 32): "bicycle",
                               (0, 0, 0): "ignore_index"}

# all cityscapes ids
CITYSCAPES_TRAINING_IDS_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 250]

# ********************************************************************
# /----- testing to assure well defined behaviour (in case values are changed by user)
# ********************************************************************

# keys should be inserted in increasing order of ids
current_key = -1
for i_n, i_c in zip(CITYSCAPES_ID_TO_NAME_16.keys(), CITYSCAPES_ID_TO_COLOR_16.keys()):
    assert i_n == i_c, 'ids be the same for name and color mapping'
    assert current_key < i_n, 'ids should be inserted into dict in increasing order'
    current_key = i_n

current_key = -1
for i_n, i_c in zip(CITYSCAPES_ID_TO_NAME_19.keys(), CITYSCAPES_ID_TO_COLOR_19.keys()):
    assert i_n == i_c, 'ids be the same for name and color mapping'
    assert current_key < i_n, 'ids should be inserted into dict in increasing order'
    current_key = i_n


def semantic_id_tensor_to_rgb_numpy_array(tensor: torch.tensor, num_classes: int):
    """
    :param num_classes: number of classes in the networks' prediction
    :param tensor: tensor of shape [1, image_height, image_width] on the cpu.
    """
    if tensor.is_cuda:
        raise Exception('Tensor on gpu please call tensor.cpu()')
    if tensor.requires_grad:
        raise Exception('Tensor requires grad please call tensor.detach()')
    id_array = tensor.numpy().transpose(1, 2, 0)
    rgb_array = np.zeros(shape=(tensor.shape[1], tensor.shape[2], 3), dtype=np.int64)

    if num_classes == 19:
        train_ids = CITYSCAPES_TRAINING_IDS_19
        id_to_color = CITYSCAPES_ID_TO_COLOR_19
    elif num_classes == 16:
        train_ids = CITYSCAPES_TRAINING_IDS_16
        id_to_color = CITYSCAPES_ID_TO_COLOR_16
    else:
        raise ValueError("Cityscapes plotting values not define for {self.num_classes} classes.")

    for cls in train_ids:
        mask = np.all(id_array == cls, axis=-1)
        rgb_array[mask] = id_to_color[cls]
    return rgb_array


def visu_depth_prediction(inv_depth_pred):
    disp_np = inv_depth_pred[:, :, :].squeeze().cpu().detach().numpy()
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_img = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_img
