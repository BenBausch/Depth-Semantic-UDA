# ------------------------------------------------------------------------------
# Simple Helper functions
# ------------------------------------------------------------------------------
import torch


def info_gpu_memory():
    """
    Prints the GPU memory (total, allocated, free) in MBytes
    1 MBytes = 1048576 Bytes
    """
    t1 = torch.cuda.get_device_properties(0).total_memory
    t2 = int(t1 % 1048576)
    t1 = int(t1 / 1048576)
    r1 = torch.cuda.memory_reserved(0)
    a1 = torch.cuda.memory_allocated(0)
    a2 = int(a1 % 1048576)
    f1 = r1 - a1  # free inside reserved
    a1 = int(a1 / 1048576)
    f2 = int(f1 % 1048576)
    f1 = int(f1 / 1048576)
    r2 = int(r1 % 1048576)
    r1 = int(r1 / 1048576)
    print(f'Total Memory: {t1}.{t2} MB, Used Memory: {a1}.{a2} MB, Free Memory: {f1}.{f2} MB, '
          f'Memory Reserved: {r1}.{r2} MB')


def sec_to_hm_str(t):
    """
    Source: https://github.com/nianticlabs/monodepth2/blob/master/utils.py
    Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def sec_to_hm(t):
    """
    Source: https://github.com/nianticlabs/monodepth2/blob/master/utils.py
    Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s
