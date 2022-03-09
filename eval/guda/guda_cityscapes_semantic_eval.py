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
        r'D:\Depth-Semantic-UDA\experiments\guda_synthia_to_cityscapes_full_depth_range\checkpoints\checkpoint_epoch_0.pth')
    for key in weights:
        if key in ['resnet_encoder', 'depth_decoder', 'semantic_decoder', 'pose_encoder', 'pose_decoder']:
            model.networks[key].load_state_dict(weights[key])
            model.networks[key].eval()
    model.eval()

    dataset = dataloaders.get_dataset(cfg.datasets.configs[2].dataset.name,
                                      'val',
                                      cfg.datasets.configs[2].dataset.split,
                                      cfg.datasets.configs[2])
    loader = DataLoader(dataset,
                        batch_size=cfg.val.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    print(f'Length of dataloader {len(loader)}')

    eval_13_classes = [True, True, True, False, False, False, True, True, True, False, True, True, True, True, False,
                       True,
                       False, True, True]
    not_eval_13_classes = [not cl for cl in eval_13_classes]
    eval_16_classes = [True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True,
                       False, True, True]
    not_eval_16_classes = [not cl for cl in eval_16_classes]

    miou_13 = MIoU(num_classes=cfg.datasets.configs[2].dataset.num_classes, ignore_classes=eval_13_classes)
    miou_16 = MIoU(num_classes=cfg.datasets.configs[2].dataset.num_classes, ignore_classes=eval_16_classes)

    for batch_idx, data in enumerate(loader):
        print(batch_idx)
        for key, val in data.items():
            data[key] = val.to('cuda:0')
        prediction = model.forward(data, dataset_id=2, predict_depth=True, train=False)[0]
        depth = prediction['depth'][0]

        sem_pred_13 = prediction['semantic']
        sem_pred_13[:, not_eval_13_classes, :, :] = 0.0

        soft_pred_13 = F.softmax(sem_pred_13, dim=1)
        miou_13.update(mask_pred=soft_pred_13, mask_gt=data['semantic'])

        sem_pred_16 = prediction['semantic']
        sem_pred_16[:, not_eval_16_classes, :, :] = 0.0

        soft_pred_16 = F.softmax(sem_pred_16, dim=1)
        miou_16.update(mask_pred=soft_pred_16, mask_gt=data['semantic'])

        rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0), caption=f'Rgb {batch_idx}')
        depth_img = get_wandb_depth_image(depth, batch_idx)
        semantic_img_13 = get_wandb_semantic_image(soft_pred_13[0], True, 1, f'Semantic Map image with 13 classes')

        semantic_img_16 = get_wandb_semantic_image(soft_pred_16[0], True, 1, f'Semantic Map image with 16 classes')
        semantic_gt = get_wandb_semantic_image(data['semantic'][0], False, 1, f'Semantic GT with id {batch_idx}')

        wandb.log({'images': [rgb_img, depth_img, semantic_img_13, semantic_img_16, semantic_gt]})

    mean_iou_13, iou_13 = miou_13.get_miou()
    mean_iou_16, iou_16 = miou_16.get_miou()

    names = [CITYSCAPES_ID_TO_NAME_19[i] for i in CITYSCAPES_ID_TO_NAME_19.keys()]
    bar_data = [[label, val] for (label, val) in zip(names, iou_13)]
    table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
    wandb.log({'IOU per 13 Class': wandb.plot.bar(table, "Classes", "IOU", title="IOU per 13 Class"),
               'Mean IOU per 13 Classes': mean_iou_13})
    names = [CITYSCAPES_ID_TO_NAME_19[i] for i in CITYSCAPES_ID_TO_NAME_19.keys()]
    bar_data = [[label, val] for (label, val) in zip(names, iou_16)]
    table = wandb.Table(data=bar_data, columns=(["Classes", "IOU"]))
    wandb.log({'IOU per 16 Class': wandb.plot.bar(table, "Classes", "IOU", title="IOU per 16 Class"),
               'Mean IOU per 16 Classes': mean_iou_16})


def get_wandb_depth_image(depth, batch_idx):
    colormapped_depth = vdp(1 / depth)
    img = wandb.Image(colormapped_depth, caption=f'Depth Map image with id {batch_idx}')
    return img


def get_wandb_semantic_image(semantic, is_prediction=True, k=1, caption=''):
    if is_prediction:
        semantic = torch.topk(semantic, k=k, dim=0, sorted=True).indices[k - 1].unsqueeze(0)
    else:
        semantic = semantic.unsqueeze(0)
    img = wandb.Image(s_to_rgb(semantic.detach().cpu()),
                      caption=caption)
    return img


if __name__ == "__main__":
    path = sys.argv[1]
    print(f'Evaluating using {path} configuration file!')
    cfg = create_configuration(path)
    evaluate_model(cfg)
