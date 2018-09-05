# visualize the prediction of branch classifiers

import numpy as np
import pickle
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
#caffe_root = '../'  
import sys
import os.path
from lenet_header import *

for net_id, useNet in enumerate(netName):
	plt.figure()
	max_acc_array=[]
	for exp_id in range(1,10):
		outputfile = caffe_dir+'outputFile/{0}_{1}_{2}.out'.format(useNet,file_suffix,exp_id)
		if not os.path.isfile(outputfile):
			print ' Missing file: ', outputfile
			continue
	
		print 'Parsing {0} exp:{1}'.format(useNet, exp_id)
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

		max_acc_array.append(max_accuracy)
		plt.subplot(3,3,exp_id)
		plt.plot(accuracy_array)
		plt.title('{}:{}'.format(exp_id,max_accuracy))
		#plt.ylabel('Accuracy')
		plt.ylim(0,1)

	plt.tight_layout(pad=1,w_pad=1,h_pad=1)

	print 'Max accuracy range: {0} ~ {1}'.format(min(max_acc_array), max(max_acc_array))
	plt.savefig('accuracyViz/accuracyViz_{0}_{1}.jpg'.format(useNet,file_suffix))
