# default python packages
import sys

# own packages
import models
import dataloaders
from cfg.config_training import create_configuration, to_dictionary
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


def evaluate_model(cfg):
    wandb.init(project=f"{cfg.experiment_name}_eval", config=to_dictionary(cfg))
    print(cfg.model.type)
    model = models.get_model(cfg.model.type, cfg)
    model = model.to('cuda:0')
    weights = torch.load(
        r'D:\Depth-Semantic-UDA\experiments\synthia_only_weighted\checkpoints\checkpoint_epoch_15.pth')
    for key in weights:
        if key in ['resnet_encoder', 'depth_decoder', 'semantic_decoder', 'pose_encoder', 'pose_decoder']:
            print(key)
            model.networks[key].load_state_dict(weights[key])
            model.networks[key].eval()
    model.eval()

    dataset = dataloaders.get_dataset(cfg.datasets.configs[1].dataset.name,
                                      'train',
                                      cfg.datasets.configs[1].dataset.split,
                                      cfg.datasets.configs[1])
    loader = DataLoader(dataset,
                        batch_size=cfg.val.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    print(f'Length of dataloader {len(loader)}')

    num_classes = cfg.datasets.configs[0].dataset.num_classes
    for i, dcfg in enumerate(cfg.datasets.configs):
        assert num_classes == dcfg.dataset.num_classes

    if num_classes == 19:
        c_id_to_name = CITYSCAPES_ID_TO_NAME_19
    elif num_classes == 16:
        c_id_to_name = CITYSCAPES_ID_TO_NAME_16
    else:
        raise ValueError("GUDA training not defined for {self.num_classes} classes!")

    eval_13_classes = [True, True, True, False, False, False, True, True, True, True, True, True, True, True,
                       True, True]

    miou_13 = MIoU(num_classes=cfg.datasets.configs[1].dataset.num_classes,
                   ignore_classes=eval_13_classes,
                   ignore_index=IGNORE_INDEX_SEMANTIC)
    miou_16 = MIoU(num_classes=cfg.datasets.configs[1].dataset.num_classes,
                   ignore_index=IGNORE_INDEX_SEMANTIC)

    for batch_idx, data in enumerate(loader):
        for key, val in data.items():
            data[key] = val.to('cuda:0')
        prediction = model.forward(data, dataset_id=1, predict_depth=True, train=False)[0]

        sem_pred_13 = prediction['semantic']

        soft_pred_13 = F.softmax(sem_pred_13, dim=1)
        miou_13.update(mask_pred=soft_pred_13, mask_gt=data['semantic'])

        sem_pred_16 = prediction['semantic']

        soft_pred_16 = F.softmax(sem_pred_16, dim=1)
        miou_16.update(mask_pred=soft_pred_16, mask_gt=data['semantic'])

        if batch_idx % 15 == 0:
            rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0),
                                  caption=f'Rgb {batch_idx}')
            semantic_img_13 = get_wandb_semantic_image(soft_pred_16[0], True, 1,
                                                            f'Semantic Map image with 16 classes')
            semantic_gt = get_wandb_semantic_image(data['semantic'][0], False, 1,
                                                        f'Semantic GT with id {batch_idx}')
            wandb.log(
                {f'images': [rgb_img, semantic_img_13, semantic_gt]})

    mean_iou_13, iou_13 = miou_13.get_miou()
    mean_iou_16, iou_16 = miou_16.get_miou()

    names = [c_id_to_name[i] for i in c_id_to_name.keys()]
    bar_data = [[label, val] for (label, val) in zip(names, iou_13)]
    table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
    wandb.log({f'IOU per 13 Class': wandb.plot.bar(table, "Classes", "IOU",
                                                                      title=f"IOU per 13 Class"),
               f'Mean IOU per 13 Classes': mean_iou_13})
    names = [c_id_to_name[i] for i in c_id_to_name.keys()]
    bar_data = [[label, val] for (label, val) in zip(names, iou_16)]
    table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
    wandb.log(
        {f'IOU per 16 Class': wandb.plot.bar(table, "Classes", "IOU",
                                                          title=f"IOU per 16 Class"),
         f'Mean IOU per 16 Classes': mean_iou_16})


def get_wandb_semantic_image(semantic, is_prediction=True, k=1, caption=''):
    if is_prediction:
        semantic = torch.topk(semantic, k=k, dim=0, sorted=True).indices[k - 1].unsqueeze(0)
    else:
        semantic = semantic.unsqueeze(0)
    img = wandb.Image(s_to_rgb(semantic.detach().cpu(), num_classes=16), caption=caption)
    return img


if __name__ == "__main__":
    path = sys.argv[1]
    print(f'Evaluating using {path} configuration file!')
    cfg = create_configuration(path)
    evaluate_model(cfg)
