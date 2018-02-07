import mxnet as mx

def ConvBnRelu(data, num_filter, kernel, stride, pad, num_group, no_bias, use_global_stats):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, no_bias=no_bias)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats)
    relu = mx.sym.Activation(data=bn, act_type='relu')
    return relu

def ConvBn(data, num_filter, kernel, stride, pad, num_group, no_bias, use_global_stats):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, no_bias=no_bias)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, use_global_stats=use_global_stats)

    return bn

def Stride2Block(data, num_filters, kernels, strides, pads, num_groups, no_bias, use_global_stats):
    conv0 = ConvBnRelu(data, num_filters[0], kernel=kernels[0], stride=strides[0], pad=pads[0], num_group=num_groups[0], no_bias=True, use_global_stats=use_global_stats)
    conv1 = ConvBn(conv0, num_filters[1], kernel=kernels[1], stride=strides[1], pad=pads[1], num_group=num_groups[1], no_bias=True, use_global_stats=use_global_stats)
    conv2 = ConvBn(conv1, num_filters[2], kernel=kernels[2], stride=strides[2], pad=pads[2], num_group=num_groups[2], no_bias=True, use_global_stats=use_global_stats)

    shortcut = mx.sym.Pooling(data, pooling_convention='full', kernel=(3,3), stride=(2,2), pool_type='avg')
    concat = mx.sym.concat(*[shortcut, conv2])
    relu = mx.sym.Activation(concat, act_type='relu')
    return relu

def ShuffleResidualBlock(data, num_filters, kernels, strides, pads, num_groups, no_bias, use_global_stats, down=False, isConcat=False):
    conv0 = ConvBnRelu(data, num_filters[0], kernel=kernels[0], stride=strides[0], pad=pads[0], num_group=num_groups[0], no_bias=True, use_global_stats=use_global_stats)
    shuffle = mx.contrib.sym.ShuffleChannel(conv0, group=3)
    conv1 = ConvBn(shuffle, num_filters[1], kernel=kernels[1], stride=strides[1], pad=pads[1], num_group=num_groups[1], no_bias=True, use_global_stats=use_global_stats)
    conv2 = ConvBn(conv1, num_filters[2], kernel=kernels[2], stride=strides[2], pad=pads[2], num_group=num_groups[2], no_bias=True, use_global_stats=use_global_stats)

    if down:
        shortcut = mx.sym.Pooling(data=data, pooling_convention='full', kernel=(3,3), stride=(2,2), pool_type='avg')
    else:
        shortcut = data

    if isConcat:
        res = mx.sym.concat(*[shortcut, conv2])
    else:
        res = data + conv2
    relu = mx.sym.Activation(data=res, act_type='relu')
    return relu


def get_symbol(num_classes, **kwargs):
    use_global_stats = False
    data = mx.sym.var(name='data')
    head = ConvBnRelu(data, num_filter=24, kernel=(3,3), stride=(2,2), pad=(1,1), num_group=1, no_bias=True, use_global_stats=use_global_stats)
    pool = mx.sym.Pooling(data=head, kernel=(3,3), pooling_convention='full', stride=(2,2), pool_type='max')
    resx1_concat_relu = Stride2Block(pool, num_filters=[54,54,216], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(2,2),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[1,1,3], no_bias=True, use_global_stats=use_global_stats)
    resx2_elewise_relu = ShuffleResidualBlock(resx1_concat_relu, num_filters=[60,60,240], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True, use_global_stats=use_global_stats)
    resx3_elewise_relu = ShuffleResidualBlock(resx2_elewise_relu, num_filters=[60,60,240], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True, use_global_stats=use_global_stats)
    resx4_elewise_relu = ShuffleResidualBlock(resx3_elewise_relu, num_filters=[60,60,240], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True, use_global_stats=use_global_stats)
    resx5_concat_relu = ShuffleResidualBlock(resx4_elewise_relu, num_filters=[60,60,240], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(2,2),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True, use_global_stats=use_global_stats, down=True, isConcat=True)
    resx6_elewise_relu = ShuffleResidualBlock(resx5_concat_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)
    resx7_elewise_relu = ShuffleResidualBlock(resx6_elewise_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)
    resx8_elewise_relu = ShuffleResidualBlock(resx7_elewise_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)
    resx9_elewise_relu = ShuffleResidualBlock(resx8_elewise_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)
    resx10_elewise_relu = ShuffleResidualBlock(resx9_elewise_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)
    resx11_elewise_relu = ShuffleResidualBlock(resx10_elewise_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)
    resx12_elewise_relu = ShuffleResidualBlock(resx11_elewise_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)
    resx13_concat_relu = ShuffleResidualBlock(resx12_elewise_relu, num_filters=[120,120,480], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(2,2),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats, down=True, isConcat=True)
    res = resx13_concat_relu
    for i in range(3):
        res = ShuffleResidualBlock(res, num_filters=[240,240,960], kernels=[(1,1),(3,3),(1,1)], strides=[(1,1),(1,1),(1,1)], pads=[(0,0),(1,1),(0,0)], num_groups=[3,1,3], no_bias=True,use_global_stats=use_global_stats)

    pool_avg = mx.sym.Pooling(data=res, kernel=(7,7), pooling_convention='full', global_pool=True, pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool_avg, name='flatten')
    fc = mx.sym.FullyConnected(flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax
