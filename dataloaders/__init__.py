from dataloaders.kitti_depricated import dataset_kitti
from dataloaders.kalimu_depricated import dataset_kalimu
from dataloaders.gta5 import dataset_gta5
from dataloaders.synthia import dataset_synthia
from dataloaders.cityscapes import dataset_cityscapes_sequence, dataset_cityscapes_semantic

available_datasets = {
    "kitti_depricated": dataset_kitti.KittiDataset,
    "kalimu_depricated": dataset_kalimu.KalimuDataset,
    "gta5": dataset_gta5.GTA5Dataset,
    "synthia_rand_cityscapes": dataset_synthia.SynthiaRandCityscapesDataset,
    "cityscapes_sequence": dataset_cityscapes_sequence.CityscapesSequenceDataset,
    "cityscapes_semantic": dataset_cityscapes_semantic.CityscapesSemanticDataset
}

def get_dataset(name_dataset, *args):
    if name_dataset not in available_datasets:
        raise KeyError("The requested dataset is not available")
    else:
        return available_datasets[name_dataset](*args)