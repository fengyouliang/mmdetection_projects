import numpy as np


class Box:
    def __init__(self, rectangle):
        '''
        rectangle class.
        :param rectangle: a list of [xmin, xmax, ymin, ymax]
        '''
        self.rec = np.array(rectangle).astype(np.int)

    @property
    def shape(self):
        '''
        get shape of Box.
        :return: shape of (height, width).
        '''
        if ((self.rec[2:] - self.rec[:2]) >= 0).all():
            wh = self.rec[2:] - self.rec[:2]
            return tuple(wh)
        else:
            return

    @property
    def area(self):
        s = self.shape
        if s is not None:
            return np.prod(s)
        else:
            return 0

    def overlap(self, other, is_iou=True):
        area1, area2 = self.area, other.area
        assert area1 > 0 and area2 > 0, 'rectangle area must be postive number.'
        rec1 = self.rec
        rec2 = other.rec
        rec1 = np.array(rec1)
        rec2 = np.array(rec2)
        top_left = np.maximum(rec1[:2], rec2[:2])
        bottom_right = np.minimum(rec1[2:], rec2[2:])
        overlap = Box([*top_left, *bottom_right]).area
        if is_iou:
            return float(overlap) / (area1 + area2 - overlap)
        else:
            return float(overlap) / area1

    def expand_by_delta(self, delta, boundary):
        xmin, ymin, xmax, ymax = self.rec
        bxmin, bymin, bxmax, bymax = boundary
        exmin = max(xmin - delta, bxmin)
        eymin = max(ymin - delta, bymin)
        exmax = min(xmax + delta, bxmax)
        eymax = min(ymax + delta, bymax)
        dt = np.array([exmin, eymin, exmax, eymax]) - self.rec
        return Box([exmin, eymin, exmax, eymax]), dt

    # def __repr__(self):
    #     print('repr')
    #     return str(self.rec)

    def __array__(self):
        print('array')
        return self.rec


if __name__ == '__main__':
    print()
    a = Box([1, 2, 3, 4])
    print()
    print(a)
    b = np.array(a)
    print()
    print(b)

    print()
