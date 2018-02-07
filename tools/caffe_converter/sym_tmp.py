import mxnet as mx
data = mx.symbol.Variable(name='data')
conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
conv1_scale = conv1_bn
conv1_relu = mx.symbol.Activation(name='conv1_relu', data=conv1_scale , act_type='relu')
resx1_conv1 = mx.symbol.Convolution(name='resx1_conv1', data=conv1_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx1_conv1_bn = mx.symbol.BatchNorm(name='resx1_conv1_bn', data=resx1_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx1_conv1_scale = resx1_conv1_bn
resx1_conv1_relu = mx.symbol.Activation(name='resx1_conv1_relu', data=resx1_conv1_scale , act_type='relu')
resx1_conv2 = mx.symbol.Convolution(name='resx1_conv2', data=resx1_conv1_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx1_conv2_bn = mx.symbol.BatchNorm(name='resx1_conv2_bn', data=resx1_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx1_conv2_scale = resx1_conv2_bn
resx1_conv2_relu = mx.symbol.Activation(name='resx1_conv2_relu', data=resx1_conv2_scale , act_type='relu')
resx1_conv3 = mx.symbol.Convolution(name='resx1_conv3', data=resx1_conv2_relu , num_filter=16, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx1_conv3_bn = mx.symbol.BatchNorm(name='resx1_conv3_bn', data=resx1_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx1_conv3_scale = resx1_conv3_bn
resx2_conv1 = mx.symbol.Convolution(name='resx2_conv1', data=resx1_conv3_scale , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx2_conv1_bn = mx.symbol.BatchNorm(name='resx2_conv1_bn', data=resx2_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx2_conv1_scale = resx2_conv1_bn
resx2_conv1_relu = mx.symbol.Activation(name='resx2_conv1_relu', data=resx2_conv1_scale , act_type='relu')
resx2_conv2 = mx.symbol.Convolution(name='resx2_conv2', data=resx2_conv1_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
resx2_conv2_bn = mx.symbol.BatchNorm(name='resx2_conv2_bn', data=resx2_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx2_conv2_scale = resx2_conv2_bn
resx2_conv2_relu = mx.symbol.Activation(name='resx2_conv2_relu', data=resx2_conv2_scale , act_type='relu')
resx2_conv3 = mx.symbol.Convolution(name='resx2_conv3', data=resx2_conv2_relu , num_filter=24, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx2_conv3_bn = mx.symbol.BatchNorm(name='resx2_conv3_bn', data=resx2_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx2_conv3_scale = resx2_conv3_bn
resx3_conv1 = mx.symbol.Convolution(name='resx3_conv1', data=resx2_conv3_scale , num_filter=144, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx3_conv1_bn = mx.symbol.BatchNorm(name='resx3_conv1_bn', data=resx3_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx3_conv1_scale = resx3_conv1_bn
resx3_conv1_relu = mx.symbol.Activation(name='resx3_conv1_relu', data=resx3_conv1_scale , act_type='relu')
resx3_conv2 = mx.symbol.Convolution(name='resx3_conv2', data=resx3_conv1_relu , num_filter=144, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx3_conv2_bn = mx.symbol.BatchNorm(name='resx3_conv2_bn', data=resx3_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx3_conv2_scale = resx3_conv2_bn
resx3_conv2_relu = mx.symbol.Activation(name='resx3_conv2_relu', data=resx3_conv2_scale , act_type='relu')
resx3_conv3 = mx.symbol.Convolution(name='resx3_conv3', data=resx3_conv2_relu , num_filter=24, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx3_conv3_bn = mx.symbol.BatchNorm(name='resx3_conv3_bn', data=resx3_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx3_conv3_scale = resx3_conv3_bn
resx3_add = mx.symbol.broadcast_add(name='resx3_add', *[resx2_conv3_scale,resx3_conv3_scale] )
resx4_conv1 = mx.symbol.Convolution(name='resx4_conv1', data=resx3_add , num_filter=144, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx4_conv1_bn = mx.symbol.BatchNorm(name='resx4_conv1_bn', data=resx4_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx4_conv1_scale = resx4_conv1_bn
resx4_conv1_relu = mx.symbol.Activation(name='resx4_conv1_relu', data=resx4_conv1_scale , act_type='relu')
resx4_conv2 = mx.symbol.Convolution(name='resx4_conv2', data=resx4_conv1_relu , num_filter=144, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
resx4_conv2_bn = mx.symbol.BatchNorm(name='resx4_conv2_bn', data=resx4_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx4_conv2_scale = resx4_conv2_bn
resx4_conv2_relu = mx.symbol.Activation(name='resx4_conv2_relu', data=resx4_conv2_scale , act_type='relu')
resx4_conv3 = mx.symbol.Convolution(name='resx4_conv3', data=resx4_conv2_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx4_conv3_bn = mx.symbol.BatchNorm(name='resx4_conv3_bn', data=resx4_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx4_conv3_scale = resx4_conv3_bn
resx5_conv1 = mx.symbol.Convolution(name='resx5_conv1', data=resx4_conv3_scale , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx5_conv1_bn = mx.symbol.BatchNorm(name='resx5_conv1_bn', data=resx5_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx5_conv1_scale = resx5_conv1_bn
resx5_conv1_relu = mx.symbol.Activation(name='resx5_conv1_relu', data=resx5_conv1_scale , act_type='relu')
resx5_conv2 = mx.symbol.Convolution(name='resx5_conv2', data=resx5_conv1_relu , num_filter=192, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx5_conv2_bn = mx.symbol.BatchNorm(name='resx5_conv2_bn', data=resx5_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx5_conv2_scale = resx5_conv2_bn
resx5_conv2_relu = mx.symbol.Activation(name='resx5_conv2_relu', data=resx5_conv2_scale , act_type='relu')
resx5_conv3 = mx.symbol.Convolution(name='resx5_conv3', data=resx5_conv2_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx5_conv3_bn = mx.symbol.BatchNorm(name='resx5_conv3_bn', data=resx5_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx5_conv3_scale = resx5_conv3_bn
resx5_add = mx.symbol.broadcast_add(name='resx5_add', *[resx4_conv3_scale,resx5_conv3_scale] )
resx6_conv1 = mx.symbol.Convolution(name='resx6_conv1', data=resx5_add , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx6_conv1_bn = mx.symbol.BatchNorm(name='resx6_conv1_bn', data=resx6_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx6_conv1_scale = resx6_conv1_bn
resx6_conv1_relu = mx.symbol.Activation(name='resx6_conv1_relu', data=resx6_conv1_scale , act_type='relu')
resx6_conv2 = mx.symbol.Convolution(name='resx6_conv2', data=resx6_conv1_relu , num_filter=192, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx6_conv2_bn = mx.symbol.BatchNorm(name='resx6_conv2_bn', data=resx6_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx6_conv2_scale = resx6_conv2_bn
resx6_conv2_relu = mx.symbol.Activation(name='resx6_conv2_relu', data=resx6_conv2_scale , act_type='relu')
resx6_conv3 = mx.symbol.Convolution(name='resx6_conv3', data=resx6_conv2_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx6_conv3_bn = mx.symbol.BatchNorm(name='resx6_conv3_bn', data=resx6_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx6_conv3_scale = resx6_conv3_bn
resx6_add = mx.symbol.broadcast_add(name='resx6_add', *[resx5_add,resx6_conv3_scale] )
resx7_conv1 = mx.symbol.Convolution(name='resx7_conv1', data=resx6_add , num_filter=144, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx7_conv1_bn = mx.symbol.BatchNorm(name='resx7_conv1_bn', data=resx7_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx7_conv1_scale = resx7_conv1_bn
resx7_conv1_relu = mx.symbol.Activation(name='resx7_conv1_relu', data=resx7_conv1_scale , act_type='relu')
resx7_conv2 = mx.symbol.Convolution(name='resx7_conv2', data=resx7_conv1_relu , num_filter=144, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx7_conv2_bn = mx.symbol.BatchNorm(name='resx7_conv2_bn', data=resx7_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx7_conv2_scale = resx7_conv2_bn
resx7_conv2_relu = mx.symbol.Activation(name='resx7_conv2_relu', data=resx7_conv2_scale , act_type='relu')
resx7_conv3 = mx.symbol.Convolution(name='resx7_conv3', data=resx7_conv2_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx7_conv3_bn = mx.symbol.BatchNorm(name='resx7_conv3_bn', data=resx7_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx7_conv3_scale = resx7_conv3_bn
resx8_conv1 = mx.symbol.Convolution(name='resx8_conv1', data=resx7_conv3_scale , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx8_conv1_bn = mx.symbol.BatchNorm(name='resx8_conv1_bn', data=resx8_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx8_conv1_scale = resx8_conv1_bn
resx8_conv1_relu = mx.symbol.Activation(name='resx8_conv1_relu', data=resx8_conv1_scale , act_type='relu')
resx8_conv2 = mx.symbol.Convolution(name='resx8_conv2', data=resx8_conv1_relu , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx8_conv2_bn = mx.symbol.BatchNorm(name='resx8_conv2_bn', data=resx8_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx8_conv2_scale = resx8_conv2_bn
resx8_conv2_relu = mx.symbol.Activation(name='resx8_conv2_relu', data=resx8_conv2_scale , act_type='relu')
resx8_conv3 = mx.symbol.Convolution(name='resx8_conv3', data=resx8_conv2_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx8_conv3_bn = mx.symbol.BatchNorm(name='resx8_conv3_bn', data=resx8_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx8_conv3_scale = resx8_conv3_bn
resx8_add = mx.symbol.broadcast_add(name='resx8_add', *[resx7_conv3_scale,resx8_conv3_scale] )
resx9_conv1 = mx.symbol.Convolution(name='resx9_conv1', data=resx8_add , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx9_conv1_bn = mx.symbol.BatchNorm(name='resx9_conv1_bn', data=resx9_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx9_conv1_scale = resx9_conv1_bn
resx9_conv1_relu = mx.symbol.Activation(name='resx9_conv1_relu', data=resx9_conv1_scale , act_type='relu')
resx9_conv2 = mx.symbol.Convolution(name='resx9_conv2', data=resx9_conv1_relu , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx9_conv2_bn = mx.symbol.BatchNorm(name='resx9_conv2_bn', data=resx9_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx9_conv2_scale = resx9_conv2_bn
resx9_conv2_relu = mx.symbol.Activation(name='resx9_conv2_relu', data=resx9_conv2_scale , act_type='relu')
resx9_conv3 = mx.symbol.Convolution(name='resx9_conv3', data=resx9_conv2_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx9_conv3_bn = mx.symbol.BatchNorm(name='resx9_conv3_bn', data=resx9_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx9_conv3_scale = resx9_conv3_bn
resx9_add = mx.symbol.broadcast_add(name='resx9_add', *[resx8_add,resx9_conv3_scale] )
resx10_conv1 = mx.symbol.Convolution(name='resx10_conv1', data=resx9_add , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx10_conv1_bn = mx.symbol.BatchNorm(name='resx10_conv1_bn', data=resx10_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx10_conv1_scale = resx10_conv1_bn
resx10_conv1_relu = mx.symbol.Activation(name='resx10_conv1_relu', data=resx10_conv1_scale , act_type='relu')
resx10_conv2 = mx.symbol.Convolution(name='resx10_conv2', data=resx10_conv1_relu , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx10_conv2_bn = mx.symbol.BatchNorm(name='resx10_conv2_bn', data=resx10_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx10_conv2_scale = resx10_conv2_bn
resx10_conv2_relu = mx.symbol.Activation(name='resx10_conv2_relu', data=resx10_conv2_scale , act_type='relu')
resx10_conv3 = mx.symbol.Convolution(name='resx10_conv3', data=resx10_conv2_relu , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx10_conv3_bn = mx.symbol.BatchNorm(name='resx10_conv3_bn', data=resx10_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx10_conv3_scale = resx10_conv3_bn
resx10_add = mx.symbol.broadcast_add(name='resx10_add', *[resx9_add,resx10_conv3_scale] )
resx11_conv1 = mx.symbol.Convolution(name='resx11_conv1', data=resx10_add , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx11_conv1_bn = mx.symbol.BatchNorm(name='resx11_conv1_bn', data=resx11_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx11_conv1_scale = resx11_conv1_bn
resx11_conv1_relu = mx.symbol.Activation(name='resx11_conv1_relu', data=resx11_conv1_scale , act_type='relu')
resx11_conv2 = mx.symbol.Convolution(name='resx11_conv2', data=resx11_conv1_relu , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
resx11_conv2_bn = mx.symbol.BatchNorm(name='resx11_conv2_bn', data=resx11_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx11_conv2_scale = resx11_conv2_bn
resx11_conv2_relu = mx.symbol.Activation(name='resx11_conv2_relu', data=resx11_conv2_scale , act_type='relu')
resx11_conv3 = mx.symbol.Convolution(name='resx11_conv3', data=resx11_conv2_relu , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx11_conv3_bn = mx.symbol.BatchNorm(name='resx11_conv3_bn', data=resx11_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx11_conv3_scale = resx11_conv3_bn
resx12_conv1 = mx.symbol.Convolution(name='resx12_conv1', data=resx11_conv3_scale , num_filter=576, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx12_conv1_bn = mx.symbol.BatchNorm(name='resx12_conv1_bn', data=resx12_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx12_conv1_scale = resx12_conv1_bn
resx12_conv1_relu = mx.symbol.Activation(name='resx12_conv1_relu', data=resx12_conv1_scale , act_type='relu')
resx12_conv2 = mx.symbol.Convolution(name='resx12_conv2', data=resx12_conv1_relu , num_filter=576, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx12_conv2_bn = mx.symbol.BatchNorm(name='resx12_conv2_bn', data=resx12_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx12_conv2_scale = resx12_conv2_bn
resx12_conv2_relu = mx.symbol.Activation(name='resx12_conv2_relu', data=resx12_conv2_scale , act_type='relu')
resx12_conv3 = mx.symbol.Convolution(name='resx12_conv3', data=resx12_conv2_relu , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx12_conv3_bn = mx.symbol.BatchNorm(name='resx12_conv3_bn', data=resx12_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx12_conv3_scale = resx12_conv3_bn
resx12_add = mx.symbol.broadcast_add(name='resx12_add', *[resx11_conv3_scale,resx12_conv3_scale] )
resx13_conv1 = mx.symbol.Convolution(name='resx13_conv1', data=resx12_add , num_filter=576, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx13_conv1_bn = mx.symbol.BatchNorm(name='resx13_conv1_bn', data=resx13_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx13_conv1_scale = resx13_conv1_bn
resx13_conv1_relu = mx.symbol.Activation(name='resx13_conv1_relu', data=resx13_conv1_scale , act_type='relu')
resx13_conv2 = mx.symbol.Convolution(name='resx13_conv2', data=resx13_conv1_relu , num_filter=576, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx13_conv2_bn = mx.symbol.BatchNorm(name='resx13_conv2_bn', data=resx13_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx13_conv2_scale = resx13_conv2_bn
resx13_conv2_relu = mx.symbol.Activation(name='resx13_conv2_relu', data=resx13_conv2_scale , act_type='relu')
resx13_conv3 = mx.symbol.Convolution(name='resx13_conv3', data=resx13_conv2_relu , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx13_conv3_bn = mx.symbol.BatchNorm(name='resx13_conv3_bn', data=resx13_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx13_conv3_scale = resx13_conv3_bn
resx13_add = mx.symbol.broadcast_add(name='resx13_add', *[resx12_add,resx13_conv3_scale] )
resx14_conv1 = mx.symbol.Convolution(name='resx14_conv1', data=resx13_add , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx14_conv1_bn = mx.symbol.BatchNorm(name='resx14_conv1_bn', data=resx14_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx14_conv1_scale = resx14_conv1_bn
resx14_conv1_relu = mx.symbol.Activation(name='resx14_conv1_relu', data=resx14_conv1_scale , act_type='relu')
resx14_conv2 = mx.symbol.Convolution(name='resx14_conv2', data=resx14_conv1_relu , num_filter=384, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
resx14_conv2_bn = mx.symbol.BatchNorm(name='resx14_conv2_bn', data=resx14_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx14_conv2_scale = resx14_conv2_bn
resx14_conv2_relu = mx.symbol.Activation(name='resx14_conv2_relu', data=resx14_conv2_scale , act_type='relu')
resx14_conv3 = mx.symbol.Convolution(name='resx14_conv3', data=resx14_conv2_relu , num_filter=160, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx14_conv3_bn = mx.symbol.BatchNorm(name='resx14_conv3_bn', data=resx14_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx14_conv3_scale = resx14_conv3_bn
resx15_conv1 = mx.symbol.Convolution(name='resx15_conv1', data=resx14_conv3_scale , num_filter=960, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx15_conv1_bn = mx.symbol.BatchNorm(name='resx15_conv1_bn', data=resx15_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx15_conv1_scale = resx15_conv1_bn
resx15_conv1_relu = mx.symbol.Activation(name='resx15_conv1_relu', data=resx15_conv1_scale , act_type='relu')
resx15_conv2 = mx.symbol.Convolution(name='resx15_conv2', data=resx15_conv1_relu , num_filter=960, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx15_conv2_bn = mx.symbol.BatchNorm(name='resx15_conv2_bn', data=resx15_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx15_conv2_scale = resx15_conv2_bn
resx15_conv2_relu = mx.symbol.Activation(name='resx15_conv2_relu', data=resx15_conv2_scale , act_type='relu')
resx15_conv3 = mx.symbol.Convolution(name='resx15_conv3', data=resx15_conv2_relu , num_filter=160, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx15_conv3_bn = mx.symbol.BatchNorm(name='resx15_conv3_bn', data=resx15_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx15_conv3_scale = resx15_conv3_bn
resx15_add = mx.symbol.broadcast_add(name='resx15_add', *[resx14_conv3_scale,resx15_conv3_scale] )
resx16_conv1 = mx.symbol.Convolution(name='resx16_conv1', data=resx15_add , num_filter=960, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx16_conv1_bn = mx.symbol.BatchNorm(name='resx16_conv1_bn', data=resx16_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx16_conv1_scale = resx16_conv1_bn
resx16_conv1_relu = mx.symbol.Activation(name='resx16_conv1_relu', data=resx16_conv1_scale , act_type='relu')
resx16_conv2 = mx.symbol.Convolution(name='resx16_conv2', data=resx16_conv1_relu , num_filter=960, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx16_conv2_bn = mx.symbol.BatchNorm(name='resx16_conv2_bn', data=resx16_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx16_conv2_scale = resx16_conv2_bn
resx16_conv2_relu = mx.symbol.Activation(name='resx16_conv2_relu', data=resx16_conv2_scale , act_type='relu')
resx16_conv3 = mx.symbol.Convolution(name='resx16_conv3', data=resx16_conv2_relu , num_filter=160, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx16_conv3_bn = mx.symbol.BatchNorm(name='resx16_conv3_bn', data=resx16_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx16_conv3_scale = resx16_conv3_bn
resx16_add = mx.symbol.broadcast_add(name='resx16_add', *[resx15_add,resx16_conv3_scale] )
resx17_conv1 = mx.symbol.Convolution(name='resx17_conv1', data=resx16_add , num_filter=960, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx17_conv1_bn = mx.symbol.BatchNorm(name='resx17_conv1_bn', data=resx17_conv1 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx17_conv1_scale = resx17_conv1_bn
resx17_conv1_relu = mx.symbol.Activation(name='resx17_conv1_relu', data=resx17_conv1_scale , act_type='relu')
resx17_conv2 = mx.symbol.Convolution(name='resx17_conv2', data=resx17_conv1_relu , num_filter=960, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
resx17_conv2_bn = mx.symbol.BatchNorm(name='resx17_conv2_bn', data=resx17_conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx17_conv2_scale = resx17_conv2_bn
resx17_conv2_relu = mx.symbol.Activation(name='resx17_conv2_relu', data=resx17_conv2_scale , act_type='relu')
resx17_conv3 = mx.symbol.Convolution(name='resx17_conv3', data=resx17_conv2_relu , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
resx17_conv3_bn = mx.symbol.BatchNorm(name='resx17_conv3_bn', data=resx17_conv3 , use_global_stats=False, fix_gamma=False, eps=0.000100)
resx17_conv3_scale = resx17_conv3_bn
conv2 = mx.symbol.Convolution(name='conv2', data=resx17_conv3_scale , num_filter=1280, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv2_bn = mx.symbol.BatchNorm(name='conv2_bn', data=conv2 , use_global_stats=False, fix_gamma=False, eps=0.000100)
conv2_scale = conv2_bn
conv2_relu = mx.symbol.Activation(name='conv2_relu', data=conv2_scale , act_type='relu')
pool_ave = mx.symbol.Pooling(name='pool_ave', data=conv2_relu , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
fc1000 = mx.symbol.Convolution(name='fc1000', data=pool_ave , num_filter=1000, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
pred = mx.symbol.SoftmaxOutput(name='pred', data=fc1000 )
