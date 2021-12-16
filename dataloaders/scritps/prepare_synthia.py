import os
import sys
from os import listdir
import shutil


def bausch_split(split_dir, base_path):
    """
    Assigns every 10th image to the validation set, no images in test set, the rest in train set.
    :param base_path: path to the folder containing the RGB, GT and Depth folders
    :param split_dir: directory of the split
    """
    ids = []
    for file_name in listdir(os.path.join(base_path, 'RGB')):
        ids.append(file_name[:-4])

    for split in ['val', 'train']:
        with open(os.path.join(split_dir, split + '.txt'), 'w') as f:
            if split == 'val':
                for i, id in enumerate(ids):
                    if i != 0 and i % 10 == 0:
                        f.write(id + '\n')
            elif split == 'train':
                for i, id in enumerate(ids):
                    if i == 0 and i % 10 != 0:
                        f.write(id + '\n')
        f.close()

    # create empty test split
    open(os.path.join(split_dir, 'test' + '.txt'), 'a').close()


def create_splits(base_path):
    """
    Creates all the split paths
    :param base_path: path to the folder containing the RGB, GT and Depth folders
    :return:
    """
    if os.path.exists(os.path.join(sys.argv[1], 'splits')):
        shutil.rmtree(os.path.join(sys.argv[1], 'splits'), ignore_errors=True)
    splits_path = os.path.join(base_path, 'splits')
    os.makedirs(splits_path)
    for split_name in available_splits.keys():
        split_dir = os.path.join(splits_path, split_name)
        os.makedirs(split_dir)
        # create the split
        available_splits[split_name](split_dir, base_path)


def create_calibration_path(base_path):
    """
    Creates the camera calibration file for the Synthia dataset.
    :param base_path: path to the folder containing the RGB, GT and Depth folders
    """
    if os.path.exists(os.path.join(sys.argv[1], 'calib')):
        shutil.rmtree(os.path.join(sys.argv[1], 'calib'), ignore_errors=True)
    calib_path = os.path.join(base_path, 'calib')
    os.makedirs(calib_path)
    with open(os.path.join(calib_path, 'calib.txt'), 'w') as f:
        calibration = "type: pinhole \nis_normalized: True \nimg_width: 544 \nimg_height: 384 \nfx: 554.254691 " \
                      "\nfy: 554.254691 \ncx: 320 \ncy: 240"
        f.write(calibration)


available_splits = {
    'bausch': bausch_split
}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Please enter base_path (folder containg the RGB, GT and Depth folders) as Argument.')
    # re/create the splits
    create_splits(sys.argv[1])
    create_calibration_path(sys.argv[1])
