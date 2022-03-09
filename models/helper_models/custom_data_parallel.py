from torch.nn.parallel import DistributedDataParallel as DDP


class CustomDistributedDataParallel(DDP):
    """DDP Module Wrapper enabling access the attributes of the underlying model"""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
