# Basic script used for evaluation a single model on a validation dataset.
# Depending on the model, I change the script if needed so this is not really a plug and play script for evaluating
# every model

# default python packages
import os
import sys

# own packages
import camera_models
import models
import dataloaders
from cfg.config_training import create_configuration, to_dictionary
from helper_modules.image_warper import _ImageToPointcloud
from io_utils import io_utils
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb, \
    CITYSCAPES_ID_TO_NAME_19
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from losses.metrics import MIoU
from utils.constans import IGNORE_INDEX_SEMANTIC
from utils.plotting_like_cityscapes_utils import CITYSCAPES_ID_TO_NAME_19, CITYSCAPES_ID_TO_NAME_16

# dependencies
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb


def evaluate_model(cfg1, cfg2):
    wandb.init(project=f"{cfg2.experiment_name}_eval", config=to_dictionary(cfg2))

    # load the guda model for the depth prediction
    model_guda = models.get_model('guda', cfg1)
    model_guda = model_guda.to('cuda:0')
    path_to_model1 = r'D:\Depth-Semantic-UDA\experiments\guda'
    model_file_name1 = r"checkpoint_epoch_113.pth"
    checkpoint1 = io_utils.IOHandler.load_checkpoint(path_to_model1, model_file_name1, 0)
    # Load pretrained weights for the model and the optimizer status
    io_utils.IOHandler.load_weights(checkpoint1, model_guda.get_networks(), None)
    model_guda.eval()

    # load the pointcloud 2 semantic model
    model_pc2s = models.get_model('pointcloud2semantic', cfg2)
    model_pc2s = model_pc2s.to('cuda:0')
    path_to_model2 = r'D:\Depth-Semantic-UDA\experiments\pc2s'
    model_file_name2 = r"checkpoint_epoch_15.pth"
    checkpoint2 = io_utils.IOHandler.load_checkpoint(path_to_model2, model_file_name2, 0)
    # Load pretrained weights for the model and the optimizer status
    io_utils.IOHandler.load_weights(checkpoint2, model_pc2s.get_networks(), None)
    model_pc2s.eval()

    # load evaluation dataset
    dataset = dataloaders.get_dataset(cfg1.datasets.configs[2].dataset.name,
                                      'val',
                                      cfg1.datasets.configs[2].dataset.split,
                                      cfg1.datasets.configs[2])
    loader = DataLoader(dataset,
                        batch_size=cfg1.val.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    print(f'Length of dataloader {len(loader)}')

    num_classes = cfg1.datasets.configs[2].dataset.num_classes
    for i, dcfg in enumerate(cfg2.datasets.configs):
        assert num_classes == dcfg.dataset.num_classes

    if num_classes == 19:
        c_id_to_name = CITYSCAPES_ID_TO_NAME_19
    elif num_classes == 16:
        c_id_to_name = CITYSCAPES_ID_TO_NAME_16
    else:
        raise ValueError("GUDA training not defined for {self.num_classes} classes!")

    eval_13_classes = [True, True, True, False, False, False, True, True, True, True, True, True, True, True,
                       True, True]

    miou_13 = MIoU(num_classes=cfg1.datasets.configs[2].dataset.num_classes,
                   ignore_classes=eval_13_classes,
                   ignore_index=IGNORE_INDEX_SEMANTIC)
    miou_16 = MIoU(num_classes=cfg1.datasets.configs[2].dataset.num_classes,
                   ignore_index=IGNORE_INDEX_SEMANTIC)

    # Set up normalized camera model for source domain
    try:
        camera_model = \
            camera_models.get_camera_model_from_file(cfg1.datasets.configs[2].dataset.camera,
                                                     os.path.join(cfg1.datasets.configs[2].dataset.path,
                                                                  "calib", "calib.txt"))

    except FileNotFoundError:
        raise Exception("No Camera calibration file found! Create Camera calibration file under: " +
                        os.path.join(cfg1.datasets.configs[2].dataset.path, "calib", "calib.txt"))

    img_width = cfg1.datasets.configs[2].dataset.feed_img_size[0]
    img_height = cfg1.datasets.configs[2].dataset.feed_img_size[1]

    camera_model = camera_model.get_scaled_model(img_width, img_height)

    image_to_pointcloud = _ImageToPointcloud(camera_model=camera_model, device=0)

    # evaluation
    for batch_idx, data in enumerate(loader):
        for key, val in data.items():
            data[key] = val.to('cuda:0')
        depth_pred = model_guda.forward(data, dataset_id=2, predict_depth=True, train=False)[0]['depth'][0][('depth', 0)].detach()
        prediction = model_pc2s.forward([image_to_pointcloud(depth_pred)])[0]

        sem_pred_13 = prediction['semantic']

        soft_pred_13 = F.softmax(sem_pred_13, dim=1)
        miou_13.update(mask_pred=soft_pred_13, mask_gt=data['semantic'])

        sem_pred_16 = prediction['semantic']

        soft_pred_16 = F.softmax(sem_pred_16, dim=1)
        miou_16.update(mask_pred=soft_pred_16, mask_gt=data['semantic'])

        rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0),
                              caption=f'Rgb {batch_idx}')
        depth_img = get_wandb_depth_image(depth_pred, batch_idx)
        semantic_img_13 = get_wandb_semantic_image(soft_pred_16[0], 16, True, 1,
                                                        f'Semantic Map image with 16 classes')
        semantic_gt = get_wandb_semantic_image(data['semantic'][0], 16, False, 1,
                                                    f'Semantic GT with id {batch_idx}')
        wandb.log(
            {f'images': [rgb_img, depth_img, semantic_img_13, semantic_gt]})


def get_wandb_depth_image(depth, batch_idx, caption_addon=''):
    """
    Creates a Wandb depth image with yellow (close) to far (black) encoding
    :param depth: depth batch, prediction or ground truth
    :param batch_idx: idx of batch in the epoch
    :param caption_addon: string
    """
    colormapped_depth = vdp(1 / depth)
    img = wandb.Image(colormapped_depth, caption=f'Depth Map of {batch_idx}' + caption_addon)
    return img


def get_wandb_semantic_image(semantic, num_classes, is_prediction=True, k=1, caption=''):
    """
    Creates a Wandb image of the semantic segmentation in cityscapes rgb encoding.
    :param semantic: semantic sample
    :param is_prediction: true if sample is a prediction
    :param k: top k prediction to be plotted
    :param caption: string
    """
    if is_prediction:
        semantic = torch.topk(semantic, k=k, dim=0, sorted=True).indices[k - 1].unsqueeze(0)
    else:
        semantic = semantic.unsqueeze(0)
    img = wandb.Image(s_to_rgb(semantic.detach().cpu(), num_classes=num_classes), caption=caption)
    return img


if __name__ == "__main__":
    path1 = sys.argv[1]
    print(f'guda depth prediction using {path1} configuration file!')
    cfg1 = create_configuration(path1)

    path2 = sys.argv[2]
    print(f'Evaluating using {path2} configuration file!')
    cfg2 = create_configuration(path2)
    evaluate_model(cfg1, cfg2)
