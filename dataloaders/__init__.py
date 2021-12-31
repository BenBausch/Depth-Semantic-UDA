from dataloaders import dataset_kitti
from dataloaders import dataset_kalimu
from dataloaders import dataset_gta5
from dataloaders import dataset_synthia

available_datasets = {
    "kitti": dataset_kitti.KittiDataset,
    "kalimu": dataset_kalimu.KalimuDataset,
    "gta5": dataset_gta5.GTA5Dataset,
    "synthia": dataset_synthia.SynthiaRandCityscapesDataset,
}

def get_dataset(name_dataset, *args):
    if name_dataset not in available_datasets:
        raise KeyError("The requested dataset is not available")
    else:
        return available_datasets[name_dataset](*args)