import numpy as np


def main(nc=5, repeat_thr=3e-1):
    # class_size = np.random.randint(50, 4000, size=nc)
    class_size = [2331, 2024, 859, 720, 567, 234, 227, 117, 108, 69]
    class_freq = class_size / np.sum(class_size)
    category_repeat = np.maximum(1.0, np.sqrt(repeat_thr / class_freq))
    print(category_repeat)
    print()


if __name__ == '__main__':
    main()
