# Package imports
import matplotlib
import numpy as np

# names of the cityscapes classes ordered by id, road has id = 0, ..., bicycle has id = 18
# the void class has id = 250 all void (non-valid) cityscapes classes will have this labels
# also classes from other datasets with no equivalent cityscapes classes e.g. synthia's Lanemarking class
import torch

CITYSCAPES_CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
                          "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                          "truck", "bus", "train", "motorcycle", "bicycle", "ignore_index"]
# used for plotting the semantic classes
# Maps cityscapes ids to specific colors, ids to names and colors to names
CITYSCAPES_ID_TO_COLOR = {0: np.asarray([128, 64, 128]),
                          1: np.asarray([244, 35, 232]),
                          2: np.asarray([70, 70, 70]),
                          3: np.asarray([102, 102, 156]),
                          4: np.asarray([190, 153, 153]),
                          5: np.asarray([153, 153, 153]),
                          6: np.asarray([250, 170, 30]),
                          7: np.asarray([220, 220, 0]),
                          8: np.asarray([107, 142, 35]),
                          9: np.asarray([152, 251, 152]),
                          10: np.asarray([0, 130, 180]),
                          11: np.asarray([220, 20, 60]),
                          12: np.asarray([255, 0, 0]),
                          13: np.asarray([0, 0, 142]),
                          14: np.asarray([0, 0, 70]),
                          15: np.asarray([0, 60, 100]),
                          16: np.asarray([0, 80, 100]),
                          17: np.asarray([0, 0, 230]),
                          18: np.asarray([119, 11, 32]),
                          250: np.asarray([0, 0, 0])}  # last class is the void class

CITYSCAPES_ID_TO_NAME = {0: "road",
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
                         250: "ignore_index"}  # last class is the void class

CITYSCAPES_COLOR_TO_NAME = {(128, 64, 128): "road",
                            (244, 35, 232): "sidewalk",
                            (70, 70, 70): "building",
                            (102, 102, 156): "wall",
                            (190, 153, 153): "fence",
                            (153, 153, 153): "pole",
                            (250, 170, 30): "traffic_light",
                            (220, 220, 0): "traffic_sign",
                            (107, 142, 35): "vegetation",
                            (152, 251, 152): "terrain",
                            (0, 130, 180): "sky",
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
CITYSCAPES_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 250]


def semantic_id_tensor_to_rgb_numpy_array(tensor: torch.tensor):
    """
    :param tensor: tensor of shape [1, image_height, image_width] on the cpu.
    """
    if tensor.is_cuda:
        raise Exception('Tensor on gpu please call tensor.cpu()')
    if tensor.requires_grad:
        raise Exception('Tensor requires grad please call tensor.detach()')
    id_array = tensor.numpy().transpose(1, 2, 0)
    rgb_array = np.zeros(shape=(tensor.shape[1], tensor.shape[2], 3), dtype=np.int64)
    for cls in CITYSCAPES_CLASS_IDS:
        mask = np.all(id_array == cls, axis=-1)
        rgb_array[mask] = CITYSCAPES_ID_TO_COLOR[cls]
    return rgb_array
