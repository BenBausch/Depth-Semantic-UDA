from models.monodepth.model_md2 import Monodepth2
from models.GUDA.model_guda import Guda
from models.deeplabV3.deepLabV3 import DeepLabV3
from models.depth2semantic.model_pc2s import PC2SModel


def get_model(model_name, *args):
    """
    """
    if model_name not in available_models:
        raise NotImplementedError('Model {} is not yet implemented'.format(model_name))
    else:
        try:
            return available_models[model_name](*args)
        except:
            raise Warning('PLease implement multiple dataset version of your model, e.g. guda!')
            return available_models[model_name](*args)


available_models = {
    'guda': Guda,
    'deeplabv3': DeepLabV3,
    'monodepth2': Monodepth2,
    'pointcloud2semantic': PC2SModel
}
