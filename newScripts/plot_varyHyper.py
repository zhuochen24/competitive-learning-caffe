# visualize the prediction of branch classifiers

import numpy as np
import pickle
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
#caffe_root = '../'  
import sys
import os.path
from lenet_header import *


plt.figure()
xvalues = np.arange(len(file_suffix_array))+0.5

for net_id, useNet in enumerate(netName):
	errorMax_acc_array=[]
	errorMin_acc_array=[]
	errorAvg_acc_array=[]

	for file_suffix in file_suffix_array:
		max_acc_array=[]
		
		for exp_id in range(1,13):
			if file_suffix == '100k_gamma01' and useNet != 'lenet_cifar10_hier3_smallBranch' and useNet != 'lenet_cifar10_hierMoE_small':
				outputfile = caffe_dir+'outputFile/{0}_{2}.out'.format(useNet,file_suffix,exp_id)
			else:
				outputfile = caffe_dir+'outputFile/{0}_{1}_{2}.out'.format(useNet,file_suffix,exp_id)
			if not os.path.isfile(outputfile):
				print ' Missing file: ', outputfile
				continue
		
			print 'Parsing {0} {2} exp:{1}'.format(useNet, exp_id, file_suffix)
			accuracy_array=[]
			max_accuracy = 0
			with open(outputfile) as f:
				for line in f:
					if 'accuracy' in line and 'Test' in line:
						#print line
						words = line.split()
						temp_accuracy = float(words[-1])
						accuracy_array.append(temp_accuracy)
						if temp_accuracy > max_accuracy:
							max_accuracy = temp_accuracy

			if max_accuracy < 0.08:
				continue
			max_acc_array.append(max_accuracy)

		print 'Max accuracy range: {0} ~ {1}'.format(min(max_acc_array), max(max_acc_array))
		errorMax_acc_array.append(max(max_acc_array)-np.mean(max_acc_array))
		errorMin_acc_array.append(np.mean(max_acc_array)-min(max_acc_array))
		errorAvg_acc_array.append(np.mean(max_acc_array))

	plt.errorbar(xvalues, errorAvg_acc_array, yerr=[errorMin_acc_array, errorMax_acc_array],label=netNameShort[net_id])

plt.ylabel('Accuracy')
#plt.xticks(xvalues, ['Gamma=0.1','Gamma=0.01','Gamma=0.001'], rotation=30)
#plt.xticks(xvalues, ['Lr=0.1','Lr=0.01','Lr=0.001'], rotation=30)
plt.xticks(xvalues, ['Gamma=2','Gamma=1','Gamma=0.1','Gamma=0.01'], rotation=30)
plt.ylim(0.,1.0)
plt.title(net_data)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(right=0.7)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.,fontsize='small')
#plt.savefig('varyGamma.jpg'.format(useNet))
#plt.savefig('varyLr.jpg'.format(useNet))
plt.savefig('lenet_MNIST_varyGamma.jpg'.format(useNet))
