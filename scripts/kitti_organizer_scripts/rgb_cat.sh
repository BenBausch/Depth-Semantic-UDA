dirs_rgb=(./*/)

rgb_dir="./rgb_data"

#
if [ -d "$rgb_dir" ]; then
  echo "The rgb directory ${rgb_dir} exists. You must delete or move it out of this folder to be able to reorganize the RGB images."
  exit 1
fi

mkdir ./$rgb_dir

echo "Organize RGB iamges..."
for dir in "${dirs_rgb[@]}"
do
    echo "Process ${dir}..."
    mv $dir/*/ ./rgb_data
done
