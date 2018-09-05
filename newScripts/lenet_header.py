
import numpy as np

caffe_dir = '/home/zhuo/caffe_python/caffe/'
batch_size_test = 100
test_img_num = 10000 #50000 for imagenet, 10000 for mnist and lenet
dataset = 'LENETMNIST'
#dataset = 'LENETCIFAR10'
#dataset = 'CONVNETCIFAR10'


if dataset=='LENETMNIST':

########## MNIST dataset ####################

	label_names = ['0','1','2','3','4','5','6','7','8','9']

	#file_suffix = 'origin'

	netName_all=['lenet_hier','lenet_hier_smallBranch','lenet_hier_small','lenet_hier_smaller','pruning']
	netSize_all = np.array([1255500.0, 50000, 39700,1985,104207])
	netSize_rel_all = netSize_all/netSize_all[0]
	netName_short_all=['Original','Small_Branch','Small','Minimal','Han et al.,15']

	numBranch_all=[2,2,2,2,2]

## experiment 1: different CNN structures
	net_data = 'LeNet on MNIST'
	file_suffix = 'gamma2'
	select = [0,1,2,3,4]
	netName = [netName_all[x] for x in select]
	netNameShort = [netName_short_all[x] for x in select]
	numBranch = [numBranch_all[x] for x in select]
	netSize_rel = [netSize_rel_all[x] for x in select]

## experiment 2: different gamma values 
#	net_data = 'LeNet on MNIST: vary gamma values'
#	file_suffix_array = ['gamma2','gamma1','gamma01','gamma001']
#	select = [0,1,2,3]
#	netName = [netName_all[x] for x in select]
#	netNameShort = [netName_short_all[x] for x in select]
#	numBranch = [numBranch_all[x] for x in select]
#	netSize_rel = [netSize_rel_all[x] for x in select]

elif dataset=='LENETCIFAR10':

########## lenet on CIFAR 10 dataset ####################
	label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
	netName_all=['lenet_cifar10_hier','lenet_cifar10_hier_small','lenet_cifar10_hierMoE_small','lenet_cifar10_hier_smallBranch','lenet_cifar10_hier_smallBclass','lenet_cifar10_hier3','lenet_cifar10_hier3_small','lenet_cifar10_hier3_smallBranch']
	netSize_all = np.array([1631500.0,52700.0,52700,58500.0,1631500,1631500,52700,58500])
	netSize_rel_all = netSize_all/netSize_all[0]
	netName_short_all=['Original','Small','Small_MoE','Small_Branch','Small_Gating','3Branches','3B_Small','3B_SmallBranch']
	numBranch_all=[2,2,2,2,2,3,3,3]

## experiment 1: different CNN structures
	net_data = 'LeNet on CIFAR10'
	#file_suffix = 'origin_lr001'
	file_suffix = '100k_gamma001'
	select = [0,1,2,3,4,5,6,7]
	netName = [netName_all[x] for x in select]
	netNameShort = [netName_short_all[x] for x in select]
	numBranch = [numBranch_all[x] for x in select]
	netSize_rel = [netSize_rel_all[x] for x in select]

### experiment 2: different gamma values 
#	net_data = 'LeNet on CIFAR10: vary Gamma'
#	file_suffix_array = ['100k_gamma01','100k_gamma001','100k_gamma0001']
#	select = [0,1,2,3,4,5,6,7]
#	netName = [netName_all[x] for x in select]
#	netNameShort = [netName_short_all[x] for x in select]
#	numBranch = [numBranch_all[x] for x in select]
#	netSize_rel = [netSize_rel_all[x] for x in select]

## experiment 3: different lr values 
#	net_data = 'LeNet on CIFAR10: vary learning rate'
#	file_suffix_array = ['100k_gamma001_lr01','100k_gamma001','100k_gamma001_lr0001']
#	select = [0,1,2,3]
#	netName = [netName_all[x] for x in select]
#	netNameShort = [netName_short_all[x] for x in select]
#	numBranch = [numBranch_all[x] for x in select]
#	netSize_rel = [netSize_rel_all[x] for x in select]

elif dataset == 'CONVNETCIFAR10':

########## convnet on CIFAR 10 dataset ####################
        label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        netName_all=['cifar10_full_hier','cifar10_full_hierMoE','cifar10_full_hier_small','cifar10_full_hier_smaller','cifar10_full_hier_smaller2','cifar10_full_hier_smaller3','cifar10_full_hier_smaller4','cifar10_full_hier_smaller5']
        netSize_all = np.array([89440.0,89440.0,48480, 25440, 13920, 17840, 9520, 7320])
        netSize_rel_all = netSize_all/netSize_all[0]
        netName_short_all=['Original','Original_MoE','Small_Branch','Smaller_Branch','Smaller_Branch2','Smaller_Branch3','Smaller_Branch4','Smaller_Branch5']
        numBranch_all=[2,2,2,2,2,2,2,2]

## experiment 1: different CNN structures
        net_data = 'ConvNet on CIFAR10'
        file_suffix = '60k_gamma1_bsize500'
        select = [0,1,2,3,4,5,6,7]
        netName = [netName_all[x] for x in select]
        netNameShort = [netName_short_all[x] for x in select]
        numBranch = [numBranch_all[x] for x in select]
        netSize_rel = [netSize_rel_all[x] for x in select]

### experiment 2: different gamma values
#       net_data = 'ConvNet on CIFAR10: vary Gamma'
#       file_suffix_array = ['60k_gamma1','60k_gamma01','60k_gamma001']
#       select = [0,1,2]
#       netName = [netName_all[x] for x in select]
#       netNameShort = [netName_short_all[x] for x in select]
#       numBranch = [numBranch_all[x] for x in select]
#       netSize_rel = [netSize_rel_all[x] for x in select]

### experiment 3: different batch sizes
#       net_data = 'ConvNet on CIFAR10: vary batch size'
#       file_suffix_array = ['60k_gamma1','60k_gamma1_bsize300','60k_gamma1_bsize500']
#       select = [0,1,2]
#       netName = [netName_all[x] for x in select]
#       netNameShort = [netName_short_all[x] for x in select]
#       numBranch = [numBranch_all[x] for x in select]
#       netSize_rel = [netSize_rel_all[x] for x in select]

