import os
import glob
import argparse
import numpy as np
import ntpath
import PIL.Image as pil
import matplotlib as mpl
from matplotlib import pyplot as plt
import cv2

# Pytorch
import torch
import dataloaders
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms, datasets

# Own classes
import models
from eval import eval
from io_utils import io_utils

# ToDo: Important: What if an image cannot be resized by keeping the aspect ratio? That case isn't covered yet...
# ToDo: Track the process, as you have no idea otherwise how long it will take to finish the testing...
# ToDo: Use the ratios also

parser = argparse.ArgumentParser(description='test_quantitative')
# parser.add_argument('--path_base_checkpoint', type=str, help='Path ewhere the checkpoints and the training configuration is stored',
#                     default='/home/petek/accepted_models/depth_estimation/2021_01_22_18_57_28_kitti')
parser.add_argument('--path_base_checkpoint', type=str, help='Path ewhere the checkpoints and the training configuration is stored',
                    default='/home/petek/accepted_models/depth_estimation/2021_03_08_18_57_49_kitti_unsupervised_no_depth_input')
parser.add_argument('--filename_checkpoint', type=str, default='checkpoint_epoch_19.pth') # ToDo: Use best here later
parser.add_argument('--no_cuda', type=bool, default=False, help='Specifies whether the test shall be executed on gpu or not')
parser.add_argument('--batch_size', type=int, default=16, help='Specifies how many images to load per test run')
parser.add_argument('--nof_workers', type=int, default=4, help = 'Specify the number of workers')
parser.add_argument('--dir_test_files', type=str, default=None)
parser.add_argument('--path_save', type=str, default='/home/petek/results/new_res_rename')
parser.add_argument('--eval_split', type=str, default=None, choices=[None, "eigen_benchmark"])
parser.add_argument('--use_gt_scale', type=bool, default=True)
parser.add_argument('--min_depth', type=float, default=0.001)
parser.add_argument('--max_depth', type=float, default=80.0)
parser.add_argument('--quantitative', type=bool, default=True)

opts = parser.parse_args()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def test(opts):
    # -----------
    # ----- SETUP
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

    # Set the eval mode as no training shall be executed here
    for m in model.get_networks().values():
        m.eval()

    # Set the path to save the results
    checkpoint_name = os.path.basename(opts.path_base_checkpoint)
    checkpoint_file = os.path.splitext(path_leaf(opts.filename_checkpoint))[0]
    path_save_results = os.path.join(opts.path_save, checkpoint_name + "_" + checkpoint_file + "_" + "all" if opts.eval_split is None else opts.eval_split )

    if not os.path.exists(path_save_results):
        os.makedirs(path_save_results)

    # Get the dataloader for the test dataset
    dataset = dataloaders.get_dataset(cfg.dataset.name, "test", opts.eval_split, cfg)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False,
                      num_workers=opts.nof_workers, pin_memory=True, drop_last=False)

    # ----------------
    # ----- PREDICTION
    print("-> Computing predictions with size {}x{}".format(
        cfg.dataset.feed_img_size[0], cfg.dataset.feed_img_size[1]))

    orig_imgs_all = []
    pred_depths_all = []
    depth_errors = []
    # ratios = []
    ctr = 0
    with torch.no_grad():
        for data in dataloader:
            print("ctr_process: ", ctr)
            ctr = ctr + 1
            # No need to resize the image here, as the dataloader takes care of that already! The images are resized
            # to the size that are trained with!
            in_img = data[("rgb", 0)].cuda()
            gt_depths = data[("gt")].cuda()
            orig_imgs_all.append(in_img)

            # ----- Predict
            pred_depths, _ = model.predict_depth(in_img)
            pred_depths_all.append(pred_depths)

            # ----- Compute depth errors
            # Go through the batch and compute the depth errors
            for i in range(pred_depths.shape[0]):
                # Get i-th prediction and i-th gt
                gt_depth = gt_depths[i].unsqueeze(0)
                gt_height, gt_width = gt_depth.shape[2], gt_depth.shape[3]
                pred_depth = pred_depths[i].unsqueeze(0)

                err_i = {}
                if opts.quantitative:
                    if opts.eval_split is not None:
                        use_garg = True if "eigen" in opts.eval_split else False
                    else:
                        use_garg = False
                    ratio_i, err_i = eval.compute_depth_losses(opts.use_gt_scale, use_garg, gt_depth,
                                                            pred_depth, (gt_height, gt_width), (opts.min_depth, opts.max_depth))
                # ratios.append(ratio_i)
                # ToDo: We do not make sure that the printing order and the order when transforming from dict to list
                #  remains the same
                err_i_list = list(err_i.values())
                depth_errors.append(err_i_list)

        pred_depths_all = torch.cat(pred_depths_all)
        orig_imgs_all = torch.cat(orig_imgs_all)

        # ------------------
        # ----- Save results
        for i in range(len(pred_depths_all)):
            print("ctr_save:", i)
            # Save the predicted, clipped, depth map
            depth_save_clipped = pred_depths_all[i].cpu().squeeze().numpy()
            depth_save_clipped = np.clip(depth_save_clipped, opts.min_depth, opts.max_depth)
            depth_save_clipped = np.uint16(depth_save_clipped * 256)
            cv2.imwrite(os.path.join(path_save_results, "{:010d}_depth_clipped.png".format(i)), depth_save_clipped)

            # Save the original depth map and the original inverse depth map
            pred_depth_np = pred_depths_all[i].squeeze().cpu().numpy()
            inv_pred_depth_np = 1.0/pred_depth_np
            path_depth = os.path.join(path_save_results, "{:010d}_depth_orig.npy".format(i))
            path_inv_depth = os.path.join(path_save_results, "{:010d}_inv_depth_orig.npy".format(i))
            np.save(path_depth, pred_depth_np)
            np.save(path_inv_depth, inv_pred_depth_np)

            # # Save the original image
            save_image(orig_imgs_all[i].cpu(), os.path.join(path_save_results, "{:010d}_orig.png".format(i)))

            # Create a proper visualization and save it
            disp_np = inv_pred_depth_np
            vmax = np.percentile(disp_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
            mapper = plt.cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_img = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
            depth_pred_visu = pil.fromarray(colormapped_img)

            path_depth_pred_visu = os.path.join(path_save_results, "{:010d}_depth_visu.jpeg".format(i))
            depth_pred_visu.save(path_depth_pred_visu)

    mean_errors = np.array(depth_errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == '__main__':
    test(opts)
