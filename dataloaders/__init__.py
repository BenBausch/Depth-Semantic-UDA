from dataloaders.gta5 import dataset_gta5
from dataloaders.synthia import dataset_synthia
from dataloaders.cityscapes import dataset_cityscapes_sequence, dataset_cityscapes_semantic
from dataloaders.kalimu import dataset_kalimu
from dataloaders.synthia_aug_cityscapes import dataset_synthia_augmented_cityscapes
from dataloaders.synthia_aug_cityscapes import dataset_synthia_cityscapes_classmix

available_datasets = {
    #"kitti_depricated": dataset_kitti.KittiDataset,
    #"kalimu_depricated": dataset_kalimu.KalimuDataset,
    "gta5": dataset_gta5.GTA5Dataset,
    "synthia_rand_cityscapes": dataset_synthia.SynthiaRandCityscapesDataset,
    "cityscapes_sequence": dataset_cityscapes_sequence.CityscapesSequenceDataset,
    "cityscapes_semantic": dataset_cityscapes_semantic.CityscapesSemanticDataset,
    'kalimu': dataset_kalimu.KalimuSequenceDataset,
    'synthia_aug_cityscapes': dataset_synthia_augmented_cityscapes.SynthiaAugCityscapesDataset,
    'synthia_cityscapes_classmix': dataset_synthia_cityscapes_classmix.SynthiaCityscapesClassMixDataset
}


def get_dataset(name_dataset, *args):
    if name_dataset not in available_datasets:
        raise KeyError("The requested dataset is not available")
    else:
        return available_datasets[name_dataset](*args)