backbone = dict(
    type='SENet',
    block='SEResNetBottleneck',
    layers=[3, 4, 6, 3],
    groups=1,
    reduction=16,
    dropout_p=None,
    inplanes=64,
    input_3x3=False,
    downsample_kernel_size=1,
    downsample_padding=0,
    num_classes=1000
)
