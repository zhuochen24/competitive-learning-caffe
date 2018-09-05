# visualize the prediction of branch classifiers

import numpy as np
import pickle
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
#caffe_root = '../'  
import sys
import os.path
from lenet_header import *

errorMax_acc_array=[]
errorMin_acc_array=[]
errorAvg_acc_array=[]

for net_id, useNet in enumerate(netName):
	plt.figure()
	max_acc_array=[]
	
	if useNet=="pruning":
		max_acc_array.append(0.989)
		errorMax_acc_array.append(max(max_acc_array)-np.mean(max_acc_array))
		errorMin_acc_array.append(np.mean(max_acc_array)-min(max_acc_array))
		errorAvg_acc_array.append(np.mean(max_acc_array))
		continue

	for exp_id in range(1,13):
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

	errorMax_acc_array.append(max(max_acc_array)-np.mean(max_acc_array))
	errorMin_acc_array.append(np.mean(max_acc_array)-min(max_acc_array))
	errorAvg_acc_array.append(np.mean(max_acc_array))


errorMax_acc_array=np.array(errorMax_acc_array)
errorMin_acc_array=np.array(errorMin_acc_array)
errorAvg_acc_array=np.array(errorAvg_acc_array)

sortedInd = np.argsort(-errorAvg_acc_array)
plt.figure()
xvalues = np.arange(len(netName))+0.5
plt.errorbar(xvalues, errorAvg_acc_array[sortedInd], yerr=[errorMin_acc_array[sortedInd], errorMax_acc_array[sortedInd]],label='Accuracy',marker='d',markersize=8)
plt.plot(xvalues,[netSize_rel[x] for x in sortedInd],marker='d',markersize=8,label='Normalized model size')
#plt.ylabel('Accuracy/Model size')
plt.xticks(xvalues, [netNameShort[x] for x in sortedInd], rotation=30,fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0,1.2)
plt.subplots_adjust(bottom=0.15)
#plt.legend(bbox_to_anchor=(0., 1.02,1.,.102),loc=3,ncol=2,mode='expand',borderaxespad=0.,fontsize='small')
plt.legend(fontsize='small')
plt.title(net_data)
plt.savefig('{0}_errorBar_{1}.jpg'.format(dataset,file_suffix))
	#print 'Max accuracy range: {0} ~ {1}'.format(min(max_acc_array), max(max_acc_array))
