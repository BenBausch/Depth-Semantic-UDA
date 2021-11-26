#!/bin/bash
train_dir="./train"
val_dir="./val"

dirs_train=(./train/*/)
dirs_val=(./val/*/)

echo "Get corresponding rgb data for training..."
for dir in "${dirs_train[@]}"
do
    # Get name of driving sequence
    arr_in=(${dir//// })
    drive_seq=${arr_in[-1]}
    echo "Process ${drive_seq}..."

    # Move rgb data into the corresponding training folder
    if [ -d ./rgb_data/$drive_seq ]; then
      mv ./rgb_data/$drive_seq/* ./train/$drive_seq/rgb/
    fi
    echo $drive_seq
done

echo "Get corresponding rgb data for validation..."
for dir in "${dirs_val[@]}"
do
    # Get name of driving sequence
    arr_in=(${dir//// })
    drive_seq=${arr_in[-1]}
    echo "Process ${drive_seq}..."

    # Move rgb data into the corresponding validation folder
    if [ -d ./rgb_data/$drive_seq ]; then
      mv ./rgb_data/$drive_seq/* ./val/$drive_seq/rgb/
    fi
    echo $drive_seq
done

