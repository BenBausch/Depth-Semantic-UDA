# python packages
import warnings
import inspect


def deprecated(func):
    """
        This decorator emits deprecation warining. If such warning is seen, please update function and remove decorator.
    """
    if inspect.isclass(func):
        warnings.warn(f'Warning!: Function {func} is depricated, please update and verify behaviour'
                      f' before using the fucntion')
        return func
    else:
        def inner(*args, **kwargs):
            warnings.warn(f'Warning!: Function {func} is depricated, please update and verify behaviour'
                          f' before using the fucntion')
            print("I can decorate any function")
            return func(*args, **kwargs)
        return inner
