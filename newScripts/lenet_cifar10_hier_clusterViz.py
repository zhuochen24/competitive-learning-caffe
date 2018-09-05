# visualize the prediction of branch classifiers

import numpy as np
import pickle
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
#caffe_root = '../'  
import sys
#sys.path.insert(0, caffe_root + 'python')
import caffe
import os.path
from lenet_header import *

caffe.set_mode_gpu()


for net_id, useNet in enumerate(netName):
	for exp_id in range(1,13):
		### for 450k run
		#if net_id == 0:
		#	caffemodel = caffe_dir+'examples/cifar10/snapshot/{0}_{1}_450k_iter_450000.caffemodel'.format(useNet,exp_id)
		#else:
		#	caffemodel = caffe_dir+'examples/cifar10/snapshot/{0}_450k_{1}_iter_450000.caffemodel'.format(useNet,exp_id)
		caffemodel = caffe_dir+'examples/cifar10/snapshot/{0}_{1}_{2}_iter_100000.caffemodel'.format(useNet,file_suffix,exp_id)
		if not os.path.isfile(caffemodel):
			print ' Missing file: ', caffemodel
			continue
		net = caffe.Net(caffe_dir+'examples/cifar10/{0}_train_test.prototxt'.format(useNet), 
				caffemodel,
				caffe.TEST)
	
		print 'Testing {0} exp:{1}'.format(useNet, exp_id)
		all_labels = np.array([])
		all_bselect = np.array([])
		for test_it in range(test_img_num/batch_size_test):
		    print '{0} out of {1}'.format(test_it,test_img_num/batch_size_test)
		    net.forward()
		    #print "======== branch classifier =========="
		    b_prob = net.blobs['smax/bclass'].data
		    print b_prob
		    print "======== Branches to Label =========="
		    labels = net.blobs['label'].data
		    all_labels = np.hstack((all_labels, labels))
		    #print labels
		    all_bselect = np.hstack((all_bselect,b_prob.argmax(-1)))
		    #branch2label = np.hstack((branch_select[:,None], labels[:,None]))

		print all_labels
		print all_bselect
		class2b0 = all_labels[all_bselect==0]
		class2b1 = all_labels[all_bselect==1]
		class2b2 = all_labels[all_bselect==2]

		plt.figure()
		plt.subplot(1,numBranch[net_id],1)
		n,bins,patches = plt.hist(class2b0)
		plt.ylabel('Branch 0')
		plt.ylim(0,1000)
		plt.xticks(bins, label_names, rotation='vertical')
		plt.subplots_adjust(bottom=0.15)

		plt.subplot(1,numBranch[net_id],2)
		n,bins,patches = plt.hist(class2b1)
		plt.ylabel('Branch 1')
		plt.ylim(0,1000)
		plt.xticks(bins, label_names, rotation='vertical')
		plt.subplots_adjust(bottom=0.15)

		if numBranch[net_id] == 3:
			plt.subplot(1,numBranch[net_id],3)
			n,bins,patches = plt.hist(class2b2)
			plt.ylabel('Branch 2')
			plt.ylim(0,1000)
			plt.xticks(bins, label_names, rotation='vertical')
			plt.subplots_adjust(bottom=0.15)

		plt.tight_layout(pad=1,w_pad=1,h_pad=1)

		plt.savefig('clusterViz/clusterViz_{0}_{1}_{2}.jpg'.format(useNet,file_suffix,exp_id))
