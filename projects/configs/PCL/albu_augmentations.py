
albu_train_transforms = [
    # HorizontalFlip, ShiftScaleRotate, RandomRotate90
    # RandomBrightnessContrast, HueSaturationValue, RGBShift
    # RandomGamma
    # CLAHE
    # Blur, MotionBlur
    # GaussNoise
    # ImageCompression
    # CoarseDropout

    dict(
        type='Cutout',
        num_holes=20,
        max_h_size=32,
        max_w_size=32,
        fill_value=0.0,
        p=0.25
    ),

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
                brightness_limit=0.1,
                contrast_limit=0.1,
                brightness_by_max=True,
                p=1.0,
            ),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=0.68,
                sat_shift_limit=0.68,
                val_shift_limit=0.1,
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