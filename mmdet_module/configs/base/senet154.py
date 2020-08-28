backbone = dict(
    type='SENet',
    block='SEBottleneck',
    layers=[3, 8, 36, 3],
    groups=64,
    reduction=16,
    dropout_p=None,
    num_classes=1000
)
