"""Create an abstract class to force all subclasses to define the methods provided in the base class
"""
import abc

class CameraModel(abc.ABC):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_image_point(self, x3d, y3d, z3d):
        pass

    @abc.abstractmethod
    def get_viewing_ray(self, pt2d):
        pass

    @abc.abstractmethod
    def image_size(self):
        pass

    @abc.abstractmethod
    def intrinsics_dict(self):
        pass

    @abc.abstractmethod
    def model_type(self):
        pass

    @abc.abstractmethod
    def fx(self):
        pass

    @abc.abstractmethod
    def fy(self):
        pass

    @abc.abstractmethod
    def cx(self):
        pass

    @abc.abstractmethod
    def cy(self):
        pass