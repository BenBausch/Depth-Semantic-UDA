from .camera_model_pinhole import *

available_camera_models = {
    "pinhole": PinholeCameraModel
    # "sick": sick
}


def get_camera_model(name_camera_model, *args):
    if name_camera_model not in available_camera_models:
        raise KeyError("The requested camera_model is not available")
    else:
        return available_camera_models[name_camera_model](*args)


def get_camera_model_from_file(name_camera_model, *args):
    if name_camera_model not in available_camera_models:
        raise KeyError("The requested camera_model is not available")
    else:
        return available_camera_models[name_camera_model].fromFile(*args)
