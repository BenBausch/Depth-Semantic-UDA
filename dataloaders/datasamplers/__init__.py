from dataloaders.datasamplers.CompletlyRandomSampler import CompletelyRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler, RandomSampler

available_samplers = {
    "CompletelyRandomSampler": CompletelyRandomSampler,
    "RandomSampler": RandomSampler,
    "SequentialSampler": SequentialSampler,
    "DistributedSampler": DistributedSampler
}


def get_sampler(name_dataset, *args, **kwargs):
    if name_dataset not in available_samplers:
        raise KeyError("The requested dataset is not available")
    else:
        return available_samplers[name_dataset](*args, **kwargs)