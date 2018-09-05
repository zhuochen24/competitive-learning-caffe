#!/bin/bash


folder='examples/cifar10'

#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id}
#done
#
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBclass 0 lenet_cifar10_template_solver.prototxt ${exp_id}
#done
#
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id}
#done


## experiment on longer epoch
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver_450k.prototxt ${exp_id} 450k
#done

## experiment with gamma=0.001

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBclass 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done


##################################################################
# Experiments based on CIFAR-10 full
# share the first three conv layers. In small, remove the third conv/relu layers
# different gamma values: 0.01, 0.1, 1
############################################################
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier_small 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma001
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma001
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hierMoE 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma001
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier_small 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hierMoE 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier_small 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma01
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma01
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hierMoE 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma01
#done

##############################################
# the original cifar10_full network with euclidean loss function. Same solver as before.
##############################################
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_eucDist 0 cifar10_full_template_solver.prototxt ${exp_id} 60k
#done

###############################
# train batch size from 100 to 500
###############################
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier_small 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hierMoE 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
#done

###############################
# train batch size from 100 to 300
###############################
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier_small 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize300
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hier 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize300
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_hierMoE 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize300
#done


###############################
## directly train single small cifar10 network, batch size=500
###############################
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh cifar10_full_small 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_origin_bsize500
#done

##################################
## different CNN sizes
###################################
for exp_id in {1..9}
do
	./${folder}/train_script.sh cifar10_full_hier_smaller 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
done
for exp_id in {1..9}
do
	./${folder}/train_script.sh cifar10_full_hier_smaller2 0 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
done
