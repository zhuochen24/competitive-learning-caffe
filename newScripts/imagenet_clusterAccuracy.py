import numpy as np

n_cluster=5
mapping_file='./clustering/{0}cluster_oneLevel.csv'.format(n_cluster)

### map both label and prediction to clusters ######
def map_to_cluster(pred, label):
    if np.array_equal(ori_label,label):
	mapping = np.genfromtxt(mapping_file,delimiter=',')
	new_label = np.copy(label)
	for ind, value in enumerate(label):
		new_label[ind] = mapping[int(value)]
	new_pred = np.copy(pred)
	for ind, value in enumerate(pred):
		new_pred[ind] = mapping[int(value)]
	return new_pred, new_label
	
### map label class to cluster
def map_class_to_cluster(label):
	mapping = np.genfromtxt(mapping_file,delimiter=',')
	new_label = np.copy(label)
	for ind, value in enumerate(label):
		new_label[ind] = int(mapping[int(value)] - 1)
	return new_label

### pick classes belonging to cluster k
def pick_n_map_class(ori_top1_pred, ori_label, k):
	mapping = np.genfromtxt(mapping_file,delimiter=',')
	new_pred = []
	new_label = []
	for pred, label in zip(ori_top1_pred, ori_label):
		#print int(label)
		#print int(mapping[int(label)])-1, k
		if int(mapping[int(label)]) == k:
			#print 'right'
			new_pred.append(int(pred))
			new_label.append(int(label))
	return new_pred, new_label

###### original 1k squeezenet. Accuracy on 9 clusters ##################
ori_label = np.genfromtxt('label_squeezenet.csv', delimiter=',')
ori_pred = np.genfromtxt('prediction_squeezenet.csv', delimiter=',')
ori_top1_pred = ori_pred.argmax(1)

mapped_top1_pred, mapped_label = map_to_cluster(ori_top1_pred, ori_label)
correct = np.sum(mapped_top1_pred == mapped_label)
print '{0} correct classification, accuracy: {1}'.format(correct, correct/float(len(ori_label)))


##### original 1k squeezenet. Accuracy per cluster ##################
print 'Accuracy of original 1k squeezenet per cluster'

mapped_top1_pred, mapped_label = map_to_cluster(ori_top1_pred, ori_label)
for k in range(n_cluster):
	temp_ind = np.where(mapped_label == k)
	correct = np.sum(mapped_top1_pred[temp_ind] == mapped_label[temp_ind])
	print 'cluster {2} || {0} correct classification among {3} samples, accuracy: {1}'.format(correct, correct/float(temp_ind[0].shape[0]), k,temp_ind[0].shape[0])

############ original 1k squeezenet on each branch ###########################
print 'Accuracy of original 1k squeezenet per branch'

for k in range(n_cluster):
	mapped_top1_pred, mapped_label = pick_n_map_class(ori_top1_pred, ori_label,k)
	mapped_label = np.array(mapped_label)
	mapped_top1_pred = np.array(mapped_top1_pred)
	correct = np.sum(mapped_top1_pred == mapped_label)
	print 'cluster: {2} || {0} correct classification among {3} samples, accuracy: {1}'.format(correct, correct/float(len(mapped_label)),k,len(mapped_label))

############ root node on 9 clusters #########################
#class_label = np.genfromtxt('label_squeezenet.csv',delimiter=',')
#ori_label = map_class_to_cluster(class_label)
##ori_label = np.genfromtxt('label_squeezenet_9cluster.csv', delimiter=',')
#ori_pred = np.genfromtxt('prediction_squeezenet_9cluster.csv', delimiter=',')
#ori_top1_pred = ori_pred.argmax(1)
#correct = np.sum(ori_top1_pred == ori_label)
#print '{0} correct classification, accuracy: {1}'.format(correct, correct/float(len(ori_label)))


##### root node accuracy per cluster ##################
#print 'Accuracy of root node per cluster'
##class_label = np.genfromtxt('label_squeezenet.csv', delimiter=',')
##ori_label = map_class_to_cluster(class_label)
#ori_label = np.genfromtxt('label_squeezenet_thin_9cluster.csv', delimiter=',')
#ori_pred = np.genfromtxt('prediction_squeezenet_thin_9cluster.csv', delimiter=',')
#ori_top1_pred = ori_pred.argmax(1)
#
#for k in range(9):
#	temp_ind = np.where(ori_label == k)
#	correct = np.sum(ori_top1_pred[temp_ind] == ori_label[temp_ind])
#	print 'cluster {2} || {0} correct classification among {3} samples, accuracy: {1}'.format(correct, correct/float(temp_ind[0].shape[0]), k,temp_ind[0].shape[0])

