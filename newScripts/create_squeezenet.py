import caffe
from caffe import layers as L, params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label 		= L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(crop_size=227, mean_value=[104,117,123]), ntop=2)
    
    n.conv1 			= L.Convolution(n.data, kernel_size=3, num_output=64, stride=2, weight_filler=dict(type='xavier'))
    n.relu1 			= L.ReLU(n.conv1, in_place=True)
    n.pool1 			= L.Pooling(n.relu1, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    #n.fire2_squeeze1x1 		= L.Convolution(n.pool1, kernel_size=1, num_output=16,weight_filler=dict(type='xavier'))
    #n.fire2_relu_squeeze1x1 	= L.ReLU(n.fire2_squeeze1x1,in_place=True)
    #n.fire2_expand1x1 		= L.Convolution(n.fire2_relu_squeeze1x1, kernel_size=1, num_output=64,weight_filler=dict(type='xavier'))
    #n.fire2_relu_expand1x1 	= L.ReLU(n.fire2_expand1x1,in_place=True)
    #n.fire2_expand3x3 		= L.Convolution(n.fire2_relu_squeeze1x1, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type='xavier'))
    #n.fire2_relu_expand3x3 	= L.ReLU(n.fire2_expand3x3,in_place=True)
    #n.fire2_concat		= L.Concat(n.fire2_relu_expand1x1,n.fire2_relu_expand3x3)

    suffix=""
    fire_num=2
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.pool1, kernel_size=1, num_output=16,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)

    fire_num=3
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire2_concat, kernel_size=1, num_output=16,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)
    n.pool3			= L.Pooling(n.fire3_concat, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    #n.fire3_squeeze1x1 		= L.Convolution(n.fire2_concat, kernel_size=1, num_output=16,weight_filler=dict(type='xavier'))
    #n.fire3_relu_squeeze1x1 	= L.ReLU(n.fire3_squeeze1x1,in_place=True)
    #n.fire3_expand1x1 		= L.Convolution(n.fire3_relu_squeeze1x1, kernel_size=1, num_output=64,weight_filler=dict(type='xavier'))
    #n.fire3_relu_expand1x1 	= L.ReLU(n.fire3_expand1x1,in_place=True)
    #n.fire3_expand3x3 		= L.Convolution(n.fire3_relu_squeeze1x1, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type='xavier'))
    #n.fire3_relu_expand3x3 	= L.ReLU(n.fire3_expand3x3,in_place=True)
    #n.fire3_concat		= L.Concat(n.fire3_relu_expand1x1,n.fire3_relu_expand3x3)
    #n.pool3			= L.Pooling(n.fire3_concat, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    
    suffix='_cluster'  # root node branch
    fire_num=4
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.pool3, kernel_size=1, num_output=8,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)
#    n.pool4_cluster			= L.Pooling(n.fire4_concat, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    fire_num=5
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire4_concat_cluster, kernel_size=1, num_output=8,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)
    exec 'n.pool{0}{1}			= L.Pooling(n.fire{0}_concat{1}, kernel_size=3, stride=2, pool=P.Pooling.MAX)'.format(fire_num,suffix)


    fire_num=6
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.pool5_cluster, kernel_size=1, num_output=12,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)

    fire_num=7
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire6_concat_cluster, kernel_size=1, num_output=12,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)

    fire_num=8
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire7_concat_cluster, kernel_size=1, num_output=16,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)

    fire_num=9
    exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire8_concat_cluster, kernel_size=1, num_output=16,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
    exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)
    exec 'n.drop{0}{1}			= L.Dropout(n.fire{0}_concat{1}, in_place=True, dropout_ratio=0.5)'.format(fire_num,suffix)

    fire_num=10
    exec 'n.conv{0}{1} 			= L.Convolution(n.fire9_concat{1}, kernel_size=1, num_output=4,weight_filler=dict(type="gaussion",mean=0.0,std=0.01))'.format(fire_num,suffix)
    #exec 'n.conv{0}_relu{1} 		= L.ReLU(n.conv{0}{1},in_place=True)'.format(fire_num,suffix)
    exec 'n.pool{0}{1}			= L.Pooling(n.conv{0}{1}, global_pooling=True, pool=P.Pooling.AVE)'.format(fire_num,suffix)
    #exec 'n.softmax{1}			= L.Softmax(n.pool{0}{1})'.format(fire_num,suffix)

    for i in range(1,num_branch+1):

        suffix='_b{0}'.format(i)  # root node branch
        fire_num=4
        exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.pool3, kernel_size=1, num_output=8,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)
#        n.pool4_cluster			= L.Pooling(n.fire4_concat, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        fire_num=5
        exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire4_concat_cluster, kernel_size=1, num_output=8,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=32,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)
        exec 'n.pool{0}{1}			= L.Pooling(n.fire{0}_concat{1}, kernel_size=3, stride=2, pool=P.Pooling.MAX)'.format(fire_num,suffix)


        fire_num=6
        exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.pool5_cluster, kernel_size=1, num_output=12,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)

        fire_num=7
        exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire6_concat_cluster, kernel_size=1, num_output=12,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=48,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)

        fire_num=8
        exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire7_concat_cluster, kernel_size=1, num_output=16,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)

        fire_num=9
        exec 'n.fire{0}_squeeze1x1{1} 	= L.Convolution(n.fire8_concat_cluster, kernel_size=1, num_output=16,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_squeeze1x1{1} 	= L.ReLU(n.fire{0}_squeeze1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand1x1{1}	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand1x1{1} 	= L.ReLU(n.fire{0}_expand1x1{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_expand3x3{1} 	= L.Convolution(n.fire{0}_relu_squeeze1x1{1}, kernel_size=3, pad=1, num_output=64,weight_filler=dict(type="xavier"))'.format(fire_num,suffix)
        exec 'n.fire{0}_relu_expand3x3{1} 	= L.ReLU(n.fire{0}_expand3x3{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.fire{0}_concat{1}		= L.Concat(n.fire{0}_relu_expand1x1{1},n.fire{0}_relu_expand3x3{1})'.format(fire_num,suffix)
        exec 'n.drop{0}{1}			= L.Dropout(n.fire{0}_concat{1}, in_place=True, dropout_ratio=0.5)'.format(fire_num,suffix)

        fire_num=10
        exec 'n.conv{0}{1} 			= L.Convolution(n.fire9_concat{1}, kernel_size=1, num_output=4,weight_filler=dict(type="gaussion",mean=0.0,std=0.01))'.format(fire_num,suffix)
        exec 'n.conv{0}_relu{1} 		= L.ReLU(n.conv{0}{1},in_place=True)'.format(fire_num,suffix)
        exec 'n.pool{0}{1}			= L.Pooling(n.conv{0}_relu{1}, global_pooling=True, pool=P.Pooling.AVE)'.format(fire_num,suffix)
        exec 'n.softmax{1}			= L.Softmax(n.pool{0}{1})'.format(fire_num,suffix)

    exec 'n.final_concat			= L.Concat(n.softmax_b1,n.softmax_b2,n.softmax_b3,n.softmax_b4)'
    exec 'n.final_concat_reshape		= L.Reshape(reshape=dict(dim=[0,-1,1000]))'
    #exec 'n.matvec				= L.Python()'
    #n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    #n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    #n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()
    
with open('multiBranch_squeezenet_train_exmaple.prototxt', 'w') as f:
    f.write(str(lenet('/home/zhuo/caffe/examples/imagenet/ilsvrc12_train_lmdb', 128)))
    
with open('multiBranch_squeezenet_test_example.prototxt', 'w') as f:
    f.write(str(lenet('/home/zhuo/caffe/examples/imagenet/ilsvrc12_train_lmdb', 25)))
