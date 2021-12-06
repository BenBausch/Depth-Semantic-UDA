from models.monodepth.model_md2 import Monodepth2
from models.packnet.model_packnet import FromMotionModelPacknet
from models.GUDA.model_guda import GudaMonodepth

def get_model(model_name, *args):
    """
    """
    if model_name not in available_models:
        raise NotImplementedError('Model {} is not yet implemented'.format(model_name))
    elif model_name == 'packnet':
        # TODO remove this if when packnet is properly implemented
        raise NotImplementedError('Packnet implementation is depricated, ' +
                                  'please update implementation to new structure of repo')
    else:
        return available_models[model_name](*args)

available_models = {
    'guda': GudaMonodepth,
    'monodepth2': Monodepth2#,
    #'packnet': FromMotionModelPacknet #TODO update packnet to new structure of repo
}