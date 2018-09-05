import numpy as np
import pickle
#import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
caffe_root = './'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(1)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('models/bvlc_reference_caffenet/snapshot/refnet_hier_nofc_1_gamma1_solver.prototxt')
#solver = caffe.SGDSolver('models/bvlc_reference_caffenet/solver.prototxt')

niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    #train_loss[it] = solver.net.blobs['loss'].data

    #print 'loss deriv',solver.net.blobs['loss'].diff[:]
    #print 'stochloss deriv',solver.net.blobs['stochloss'].diff[:]
    #print 'conv1 deriv', solver.net.params['conv1'][0].diff[:]
    #print 'stochloss deriv[0] out/class',solver.net.blobs['stochloss'].diff[:]
    #print 'stochloss deriv[2] bclass', solver.net.blobs['stochloss'].diff[:]
    print 'fc8/b0 deriv', solver.net.params['fc8/b0'][0].diff[:]
    print 'fc8/b1 deriv', solver.net.params['fc8/b1'][0].diff[:]


    #print 'fc8', solver.net.params['fc8'][0].diff[:]

    
   # # store the output on the first test batch
   # # (start the forward pass at conv1 to avoid loading new data)
   # solver.test_nets[0].forward(start='conv1')
   # output[it] = solver.test_nets[0].blobs['score'].data[:8]
    
   # # run a full test every so often
   # # (Caffe can also do this for us and write to a log, but we show here
   # #  how to do it directly in Python, where more complicated things are easier.)
   # if it % test_interval == 0:
   #     print 'Iteration', it, 'testing...'
   #     correct = 0
   #     for test_it in range(100):
   #         solver.test_nets[0].forward()
   #         correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
   #                        == solver.test_nets[0].blobs['label'].data)
   #     test_acc[it // test_interval] = correct / 1e4


#batch_size_test = 25
#test_img_num = 50000
#netName = {0:'VGG16', 1:'SQUEEZENET', 2:'ROOT', 3:'THIN_ROOT'}
#useNet = netName[3]
#
#def save_for_cluster(pred, label, pred_file, label_file):
#    with open(label_file,'a') as f:
#    	np.savetxt(f,label,delimiter=",")
#    with open(pred_file,'a') as f:
#    	np.savetxt(f,pred,delimiter=",")
#
##caffe.set_device(0)
#caffe.set_mode_gpu()
#if useNet == 'VGG16':
#	net = caffe.Net('models/vgg/my_vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt', 
#			'models/vgg/my_vgg_16/VGG_ILSVRC_16_layers.caffemodel',
#			caffe.TEST)
#elif useNet == 'SQUEEZENET':
#	net = caffe.Net('SqueezeNet/SqueezeNet_v1.1/train_val.prototxt', 
#			'SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel',
#			caffe.TEST)
#elif useNet == 'ROOT':
#	net = caffe.Net('SqueezeNet/SqueezeNet_v1.1/train_val_cluster.prototxt', 
#			'records/squeezenet_cluster1_retrain/train_iter_20000.caffemodel',
#			caffe.TEST)
#elif useNet == 'THIN_ROOT':
#	net = caffe.Net('SqueezeNet/SqueezeNet_v1.1/train_val_cluster_small.prototxt', 
#			'records/squeezenet_cluster_small6/train_iter_340000.caffemodel',
#			caffe.TEST)
##print [(k, v.data.shape) for k, v in net.blobs.items()]
##print [(k, v[0].data.shape) for k, v in net.params.items()]
#
#all_top1_prediction=[]
#all_true_label = []
#print 'Testing...'
#correct = 0
#for test_it in range(test_img_num/batch_size_test):
#    print '{0} out of {1}'.format(test_it,test_img_num/batch_size_test)
#    net.forward()
#    #print test_net.blobs['ip2'].data.argmax(1)
#    #print np.squeeze(test_net.blobs['label'].data)
#    #prediction = net.blobs['prob'].data # vgg 16
#    prediction = net.blobs['pool10'].data  # squeezenet
#    true_label = np.squeeze(net.blobs['label'].data)
#    correct += np.sum(prediction.argmax(1) == true_label)
#
#    #append to file for clustering
#    save_for_cluster(prediction, true_label,'prediction_squeezenet_thin_9cluster.csv','label_squeezenet_thin_9cluster.csv')
#
#print 'correct: {0}; accuracy: {1}'.format(correct, correct/float(test_img_num))



