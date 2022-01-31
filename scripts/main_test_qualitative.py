import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
from matplotlib import pyplot as plt

# Pytorch
import torch
from torchvision import transforms, datasets

# Own classes
import models
from io_utils import io_utils

parser = argparse.ArgumentParser(description='test_visu')
parser.add_argument('--path_imgs', type=str, help='Path to an RGB image or directory with RGB images to test with',
                    default= "/home/petek/Desktop/test")
parser.add_argument('--path_checkpoint', type=str, help='Path to the checkpoint to be used for loading pretrained weights',
                    default='/home/petek/accepted_models/depth_estimation/2021_01_22_18_57_28_kitti/checkpoints/checkpoint_epoch_19.pth')
parser.add_argument('--path_base_checkpoint', type=str, help='Path where the checkpoints and the training configuration is stored',
                    default='/home/petek/accepted_models/depth_estimation/2021_01_22_18_57_28_kitti')
parser.add_argument('--filename_checkpoint', type=str, default='checkpoint_epoch_19.pth') # ToDo: Use best here later
parser.add_argument('--img_extension', type=str, help='Extension of images to test with',
                    default='png')
parser.add_argument('--no_cuda', type=bool, default=False, help='Specifies whether the test shall be executed on gpu or not')

opts = parser.parse_args()

def test(opts):
    # Specify the device
    device = torch.device("cuda" if not opts.no_cuda else "cpu")
    torch.backends.cudnn.benchmark = True if not opts.no_cuda else False

    # Load the checkpoint
    checkpoint = io_utils.IOHandler.load_checkpoint(opts.path_base_checkpoint, opts.filename_checkpoint)

    # Load the configuration
    path_cfg = os.path.join(opts.path_base_checkpoint, 'cfg.yaml')
    assert os.path.exists(path_cfg), "The path {} does not exist!".format(path_cfg)
    cfg = io_utils.IOHandler.load_cfg(path_cfg)

    # Create model without pretrained weights
    model = models.get_model(cfg.model.type, device, cfg)

    # Load pretrained weights and other variables stored in the checkpoint
    io_utils.IOHandler.load_weights(checkpoint, model.get_networks())
    feed_img_width = cfg.dataset.feed_img_size[0]
    feed_img_height = cfg.dataset.feed_img_size[1]

    # Set the eval_functions mode as no training shall be executed here
    for m in model.get_networks().values():
        m.eval()

    # Check whether the path refers to an image or a directory of images
    if os.path.isfile(opts.path_imgs):
        # Only testing on a single image
        img_paths = [opts.path_imgs]
        output_directory = os.path.dirname(opts.path_imgs)
    elif os.path.isdir(opts.path_imgs):
        # Searching folder for images
        img_paths = glob.glob(os.path.join(opts.path_imgs, '*.{}'.format(opts.img_extension)))
        output_directory = opts.path_imgs
    else:
        raise Exception("Can not find opts.path_imgs: {}".format(opts.path_imgs))
    print("-> Predicting on {:d} test images".format(len(img_paths)))

    with torch.no_grad():
        for idx, image_path in enumerate(img_paths):
            if image_path.endswith("_pred.jpeg"):
                continue

            # Load image
            in_img = pil.open(image_path).convert('RGB')

            # Transform the image, i.e. resize to the feed_size while training and transform to tensor
            orig_width, orig_height = in_img.size
            in_img = in_img.resize((feed_img_width, feed_img_height), pil.LANCZOS)
            in_img = transforms.ToTensor()(in_img).unsqueeze(0)
            in_img = in_img.to(device)

            # Prediction
            depth_pred, _ = model.predict_depth(in_img)
            inv_depth_pred = 1 / depth_pred

            # Save the predicted depth map and the inverse depth map (numpy files)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            path_depth = os.path.join(output_directory, "{}_depth.npy".format(output_name))
            path_inv_depth = os.path.join(output_directory, "{}_inv_depth.npy".format(output_name))
            np.save(path_depth, depth_pred.cpu().numpy())
            np.save(path_inv_depth, inv_depth_pred.cpu().numpy())

            # Create a proper visualization
            disp_np = inv_depth_pred[:, :, :, :].squeeze().cpu().detach().numpy()
            vmax = np.percentile(disp_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
            mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_img = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
            depth_pred_visu = pil.fromarray(colormapped_img)

            path_depth_pred_visu = os.path.join(output_directory, "{}_depth_visu.jpeg".format(output_name))
            depth_pred_visu.save(path_depth_pred_visu)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(img_paths), path_depth_pred_visu))

    print("Done.")


if __name__ == '__main__':
    test(opts)
