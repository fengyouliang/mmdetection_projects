wheat_top1_transforms = [
    # HorizontalFlip, ShiftScaleRotate, RandomRotate90
    # RandomBrightnessContrast, HueSaturationValue, RGBShift
    # RandomGamma
    # CLAHE
    # Blur, MotionBlur
    # GaussNoise
    # ImageCompression
    # CoarseDropout

    # HorizontalFlip, ShiftScaleRotate, RandomRotate90
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HorizontalFlip',
                p=1.0,
            ),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                value=None,
                mask_value=None,
                p=1.0,
            ),
            dict(
                type='RandomRotate90',
                p=1.0,
            ),
        ], p=0.5),

    # InvertImg
    dict(
        type='InvertImg',
        p=0.5,
    ),

    # RandomBrightnessContrast, HueSaturationValue, RGBShift
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.2,
                contrast_limit=0.2,
                brightness_by_max=True,
                p=1.0,
            ),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0,
            ),
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0,
            ),
        ], p=0.5),

    # RandomGamma
    dict(
        type='RandomGamma',
        gamma_limit=(80, 120),
        eps=None,
        p=0.5,
    ),

    # CLAHE
    dict(
        type='CLAHE',
        clip_limit=4.0,
        tile_grid_size=(8, 8),
        p=0.5,
    ),

    # Blur, MotionBlur
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='Blur',
                blur_limit=7,
                p=1.0,
            ),
            dict(
                type='MotionBlur',
                blur_limit=7,
                p=1.0,
            ),
        ], p=0.5),

    # GaussNoise
    dict(
        type='GaussNoise',
        var_limit=(10.0, 50.0),
        mean=0,
        p=0.5,
    ),

    # ImageCompression
    dict(
        type='ImageCompression',
        quality_lower=99,
        quality_upper=100,
        p=0.5,
    ),

    # CoarseDropout
    dict(
        type='CoarseDropout',
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        fill_value=0,
        p=0.5,
    ),
]

default_albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

kaggle_albu = [
    dict(type='ToFloat', max_value=255.0),
    dict(
        type='RandomSizedCrop',
        min_max_height=(650, 1024),
        height=1024,
        width=1024,
        p=0.5),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=0.68,
        sat_shift_limit=0.68,
        val_shift_limit=0.1,
        p=0.75),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.33),
    dict(type='RandomRotate90', p=0.5),
    dict(
        type='Cutout',
        num_holes=20,
        max_h_size=32,
        max_w_size=32,
        fill_value=0.0,
        p=0.25),
    dict(type='FromFloat', max_value=255.0, dtype='uint8')
]

