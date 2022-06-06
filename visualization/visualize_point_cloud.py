import os
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import camera_models
from cfg.config_training import create_configuration
from helper_modules.image_warper import _ImageToPointcloud

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from camviz.draw.draw import Draw
from camviz.objects import Camera
from camviz.utils.cmaps import jet

from io_utils import io_utils
from utils.plotting_like_cityscapes_utils import semantic_id_tensor_to_rgb_numpy_array as s_to_rgb
import dataloaders
import models


if __name__ == "__main__":
    print('SET DOFLIP FALSE !!!')
    path_to_run_yaml = r"C:\Users\benba\Documents\University\Masterarbeit\Depth-Semantic-UDA\cfg\yaml_files\plot\plot_guda_synthia_cityscapes.yaml"
    path_to_model_weights = r"C:\Users\benba\Documents\University\Masterarbeit\models\ema_stuff"
    model_name = r"checkpoint_epoch_9.pth"
    dataset_id = 1

    cfg = create_configuration(path_to_run_yaml)
    dataset_cfg = cfg.datasets.configs[dataset_id]

    # load the model
    model = models.get_model(cfg.model.type, cfg).cuda()
    checkpoint = io_utils.IOHandler.load_checkpoint(path_to_model_weights, model_name, 0)
    io_utils.IOHandler.load_weights(checkpoint, model.get_networks(), None)
    for m in model.networks.values():
        m.eval()

    # load the dataset
    dataset = dataloaders.get_dataset(dataset_cfg.dataset.name,
                                      'train',
                                      dataset_cfg.dataset.split,
                                      dataset_cfg)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

    # get camera
    my_camera = camera_models.get_camera_model_from_file(dataset_cfg.dataset.camera,
                                                      os.path.join(dataset_cfg.dataset.path, "calib",
                                                                   "calib.txt"))
    my_camera = my_camera.get_scaled_model(dataset_cfg.dataset.feed_img_size[0], dataset_cfg.dataset.feed_img_size[1])
    intrinsics = torch.Tensor([[my_camera.fx(), 0.,             my_camera.cx()],
                               [0.,             my_camera.fy(), my_camera.cy()],
                               [0.,             0.,             1.]])

    print(intrinsics)

    img2pc = _ImageToPointcloud(my_camera, 'cuda:0')

    # predict semantic and depth

    # Create draw tool with specific width and height window dimensions
    draw = Draw(wh=(2000, 900), title=f'Prediction on {dataset_cfg.dataset.name}')

    # Create image screen to show the RGB image
    draw.add2Dimage('rgb', luwh=(0.00, 0.00, 0.33, 0.50), res=dataset_cfg.dataset.feed_img_size)

    # Create image screen to show the depth visualization
    draw.add2Dimage('viz', luwh=(0.00, 0.50, 0.33, 1.00), res=dataset_cfg.dataset.feed_img_size)

    # Create world screen at specific position inside the window (% left/up/right/down)
    draw.add3Dworld('wld', luwh=(0.33, 0.00, 1.00, 1.00),
                    pose=(7.25323, -3.80291, -5.89996, 0.98435, 0.07935, 0.15674, 0.01431))

    for idx, data in enumerate(loader):

        for key, val in data.items():
            data[key] = val.cuda()

        prediction = model.forward(data, dataset_id=dataset_id, train=False)[0]
        semantic = F.softmax(prediction['semantic'], dim=1)[0]
        semantic = torch.topk(semantic, k=1, dim=0, sorted=True).indices[0].unsqueeze(0)
        semantic = s_to_rgb(semantic.detach().cpu(), num_classes=dataset_cfg.dataset.num_classes)

        rgb = data[('rgb', 0)][0] + torch.tensor([[[0.28689554, 0.32513303, 0.28389177]]]).transpose(0, 2).cuda()
        rgb = rgb.detach().cpu().numpy().transpose(1, 2, 0)

        depth = prediction['depth'][0][('depth', 0)][0].cpu().detach().numpy().squeeze(0)
        #pc = img2pc(prediction['depth'][0][('depth', 0)])[0].cpu().detach().numpy().reshape(256*512, 2)

        # Get image resolution
        wh = rgb.shape[:2][::-1]

        # Parse dictionary information
        viz = semantic / 255 # needs to

        # Create camera from intrinsics and image dimensions (width and height)
        camera = Camera(K=intrinsics, wh=wh)

        # Project depth maps from image (i) to camera (c) coordinates
        points = camera.i2c(depth)
        print(points)

        # Create pointcloud colors
        rgb_clr = rgb.reshape(-1, 3)  # RGB colors
        viz_clr = semantic.reshape(-1, 3) # Depth visualization colors

        # Create RGB and visualization textures
        draw.addTexture('rgb', rgb)  # Create texture buffer to store rgb image
        draw.addTexture('viz', viz)  # Create texture buffer to store visualization image

        # Create buffers to store data for display
        draw.addBufferf('pts', points)  # Create data buffer to store depth points
        draw.addBufferf('clr', rgb_clr)  # Create data buffer to store rgb points color
        draw.addBufferf('viz', viz_clr)  # Create data buffer to store pointcloud heights

        # Color dictionary
        color_dict = {0: 'clr', 1: 'viz'}

        # Display loop
        color_mode = 0
        while draw.input():
            # If RETURN is pressed, switch color mode
            if draw.RETURN:
                color_mode = (color_mode + 1) % len(color_dict)
            # Clear window
            draw.clear()
            # Draw image textures on their respective screens
            draw['rgb'].image('rgb')
            draw['viz'].image('viz')
            # Draw points and colors from buffer
            draw['wld'].size(2).points('pts', color_dict[color_mode])
            # Draw camera with texture as image
            draw['wld'].object(camera, tex='rgb')
            # Update window
            draw.update(30)
            draw.save(fr'C:\Users\benba\Desktop\pc\{idx}.png')
