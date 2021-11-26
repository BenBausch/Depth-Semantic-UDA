from models.monodepth.model_md2 import ModelMonodepth2
from models.packnet.model_packnet import ModelPacknet

def get_model(model_name, *args):
    """
    """
    if model_name not in available_models:
        raise NotImplementedError('Model {} is not yet implemented'.format(model_name))
    else:
        return available_models[model_name](*args)

available_models = {
    'monodepth2': ModelMonodepth2,
    'packnet': ModelPacknet
}