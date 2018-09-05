import numpy as np
import pickle
#import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
caffe_root = './'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

batch_size_test = 25
test_img_num = 50000
netName = {0:'VGG16', 1:'SQUEEZENET', 2:'ROOT', 3:'THIN_ROOT'}
useNet = netName[3]

def save_for_cluster(pred, label, pred_file, label_file):
    with open(label_file,'a') as f:
    	np.savetxt(f,label,delimiter=",")
    with open(pred_file,'a') as f:
    	np.savetxt(f,pred,delimiter=",")

#caffe.set_device(0)
caffe.set_mode_gpu()
if useNet == 'VGG16':
	net = caffe.Net('models/vgg/my_vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt', 
			'models/vgg/my_vgg_16/VGG_ILSVRC_16_layers.caffemodel',
			caffe.TEST)
elif useNet == 'SQUEEZENET':
	net = caffe.Net('SqueezeNet/SqueezeNet_v1.1/train_val.prototxt', 
			'SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel',
			caffe.TEST)
elif useNet == 'ROOT':
	net = caffe.Net('SqueezeNet/SqueezeNet_v1.1/train_val_cluster.prototxt', 
			'records/squeezenet_cluster1_retrain/train_iter_20000.caffemodel',
			caffe.TEST)
elif useNet == 'THIN_ROOT':
	net = caffe.Net('SqueezeNet/SqueezeNet_v1.1/train_val_cluster_small.prototxt', 
			'records/squeezenet_cluster_small6/train_iter_340000.caffemodel',
			caffe.TEST)
#print [(k, v.data.shape) for k, v in net.blobs.items()]
#print [(k, v[0].data.shape) for k, v in net.params.items()]

all_top1_prediction=[]
all_true_label = []
print 'Testing...'
correct = 0
for test_it in range(test_img_num/batch_size_test):
    print '{0} out of {1}'.format(test_it,test_img_num/batch_size_test)
    net.forward()
    #print test_net.blobs['ip2'].data.argmax(1)
    #print np.squeeze(test_net.blobs['label'].data)
    #prediction = net.blobs['prob'].data # vgg 16
    prediction = net.blobs['pool10'].data  # squeezenet
    true_label = np.squeeze(net.blobs['label'].data)
    correct += np.sum(prediction.argmax(1) == true_label)

    #append to file for clustering
    save_for_cluster(prediction, true_label,'prediction_squeezenet_thin_9cluster.csv','label_squeezenet_thin_9cluster.csv')

print 'correct: {0}; accuracy: {1}'.format(correct, correct/float(test_img_num))



