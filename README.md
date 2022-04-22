# Setting Up the environement

- Install miniconda if not installed already
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh (hit enter, yes, select your directory)
  bash
- conda create -n Masterarbeit python=3.8
- conda activate Masterarbeit 
- python --version --> should return python 3.8.12
- pip install -r requirements.txt
- conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
- todo 
- see cluster_scripts on slurm
- 
