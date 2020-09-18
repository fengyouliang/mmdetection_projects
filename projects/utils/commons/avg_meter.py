import numpy as np


class AverageMeter(object):
    def __init__(self, size=1, dtype=np.float64):
        self.size = size
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.val = np.zeros(self.size, dtype=self.dtype)
        self.avg = np.zeros(self.size, dtype=self.dtype)
        self.sum = np.zeros(self.size, dtype=self.dtype)
        self.count = np.zeros(self.size, dtype=self.dtype)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        assert abs(self.avg.sum() - 1) < 1e-15

    def __repr__(self):
        return '\t'.join(self.avg.astype(str).tolist())


def check_avg_meter():
    pass

