#!/bin/bash
train_dir="./train"
val_dir="./val"

dirs_train=(./data_depth_velodyne/train/*/)
dirs_val=(./data_depth_velodyne/val/*/)

# Check whether directory already exists. Exit if it does, as it should not be modified in that case.
if [ -d "$train_dir" ]; then
  echo "The train directory ${train_dir} exists. You must delete or move it out of this folder to be able to reorganize the KITTI dataset."
  exit 1
fi

# Don't like this...
if [ -d "$val_dir" ]; then
  echo "The validation directory ${val_dir} exists. You must delete or move it out of this folder to be able to reorganize the KITTI dataset."
  exit 1
fi

mkdir $train_dir
mkdir $val_dir

echo "Organize train data..."
for dir in "${dirs_train[@]}"
do
    # Get name of driving sequence
    arr_in=(${dir//// })
    drive_seq=${arr_in[-1]}
    echo "Process ${drive_seq}..."

    # Create directories
    mkdir -p ./train/$drive_seq/rgb
    mkdir -p ./train/$drive_seq/velodyne_raw/proj_depth
    mkdir -p ./train/$drive_seq/groundtruth/proj_depth

    # Move data into newly created directories
    if [ -d ./rgb_data/$drive_seq ]; then
      mv ./rgb_data/$drive_seq/* ./train/$drive_seq/rgb/
    fi
    if [ -d ./data_depth_velodyne/train/$drive_seq ]; then
      mv ./data_depth_velodyne/train/$drive_seq/proj_depth/velodyne_raw/* ./train/$drive_seq/velodyne_raw/proj_depth/
    fi
    if [ -d ./data_depth_annotated/train/$drive_seq ]; then
      mv ./data_depth_annotated/train/$drive_seq/proj_depth/groundtruth/* ./train/$drive_seq/groundtruth/proj_depth/
    fi
    echo $drive_seq
done


# take the if clauses out later...
echo "Organize validation data..."
for dir in "${dirs_val[@]}"
do
    # Get name of driving sequence
    arr_in=(${dir//// })
    drive_seq=${arr_in[-1]}
    echo "Process ${drive_seq}..."

    # Create directories
    mkdir -p ./val/$drive_seq/rgb
    mkdir -p ./val/$drive_seq/velodyne_raw/proj_depth
    mkdir -p ./val/$drive_seq/groundtruth/proj_depth

    # Move data into newly created directories
    if [ -d ./rgb_data/$drive_seq ]; then
      mv ./rgb_data/$drive_seq/* ./val/$drive_seq/rgb/
    fi
    if [ -d ./data_depth_velodyne/val/$drive_seq ]; then
      mv ./data_depth_velodyne/val/$drive_seq/proj_depth/velodyne_raw/* ./val/$drive_seq/velodyne_raw/proj_depth/
    fi
    if [ -d ./data_depth_annotated/val/$drive_seq ]; then
      mv ./data_depth_annotated/val/$drive_seq/proj_depth/groundtruth/* ./val/$drive_seq/groundtruth/proj_depth/
    fi
    echo $drive_seq
done

echo "Done"
