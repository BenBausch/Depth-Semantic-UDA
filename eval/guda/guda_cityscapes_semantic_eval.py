# default python packages
import sys

# own packages
import models
import dataloaders
from cfg.config_training import create_configuration, to_dictionary
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
from utils.plotting_like_cityscapes_utils import visu_depth_prediction as vdp
from losses.metrics import MIoU

# dependencies
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb


def evaluate_model(cfg):
    #wandb.init(project=f"{cfg.experiment_name}_eval", config=to_dictionary(cfg))
    print(cfg.model.type)
    model = models.get_model(cfg.model.type, cfg)
    model = model.to('cuda:0')
    weights = torch.load(r'C:\Users\benba\Documents\University\Masterarbeit\models\guda_model\checkpoints\checkpoint_epoch_2.pth')
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

    miou = MIoU(num_classes=cfg.datasets.configs[2].dataset.num_classes)

    for batch_idx, data in enumerate(loader):
        for key, val in data.items():
            data[key] = val.to('cuda:0')
        prediction = model.forward(data, dataset_id=2, predict_depth=True, train=False)[0]
        depth = prediction['depth'][0]

        soft_pred = F.softmax(prediction['semantic'], dim=1)
        miou.update(mask_pred=soft_pred, mask_gt=data['semantic'])

        rgb_img = wandb.Image(data[('rgb', 0)][0].cpu().detach().numpy().transpose(1, 2, 0), caption=f'Rgb {batch_idx}')
        depth_img = get_wandb_depth_image(depth, batch_idx)
        semantic_img = get_wandb_semantic_image(soft_pred[0], True, 1, f'Semantic Map image with id {batch_idx}')
        semantic2_img = get_wandb_semantic_image(soft_pred[0], True, 2, f'Semantic Map k=2 image with id {batch_idx}')
        semantic_gt = get_wandb_semantic_image(data['semantic'][0], False, 1, f'Semantic GT with id {batch_idx}')


        #wandb.log({'images': [rgb_img, depth_img, semantic_img, semantic2_img, semantic_gt]})

    mean_iou, iou = miou.get_miou()
    print(iou)
    print(mean_iou)


def get_wandb_depth_image(depth, batch_idx):
    colormapped_depth = vdp(1/depth)
    img = wandb.Image(colormapped_depth, caption=f'Depth Map image with id {batch_idx}')
    return img


def get_wandb_semantic_image(semantic, is_prediction=True, k=1, caption=''):
    if is_prediction:
        semantic = torch.topk(semantic, k=k, dim=0, sorted=True).indices[k-1].unsqueeze(0)
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