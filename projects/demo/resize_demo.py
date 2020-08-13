import numpy as np


def random_sample(img_scales):
    img_scale_long = [max(s) for s in img_scales]
    img_scale_short = [min(s) for s in img_scales]
    long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
    short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
    img_scale = (long_edge, short_edge)
    return img_scale, None


def random_select(img_scales):
    scale_idx = np.random.randint(len(img_scales))
    img_scale = img_scales[scale_idx]
    return img_scale, scale_idx


if __name__ == '__main__':
    print(random_sample([(1333, 800), (640, 480)]))
    print(random_select([(1333, 800), (1333, 640)]))
