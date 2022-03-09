from models.monodepth.model_md2 import Monodepth2
from models.GUDA.model_guda import Guda
from models.deeplabV2_dada.deepLabV2Depth import DeepLabV2DADA

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
    'dada_modif': DeepLabV2DADA,
    'monodepth2': Monodepth2
}
