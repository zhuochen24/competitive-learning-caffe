#!/bin/bash


folder='examples/cifar10'

## experiment on different network sizes
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3 1 lenet_cifar10_template_solver.prototxt ${exp_id}
#done
#
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_small 1 lenet_cifar10_template_solver.prototxt ${exp_id}
#done
#
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier 1 lenet_cifar10_template_solver.prototxt ${exp_id}
#done

## experiment on longer epoch
#for exp_id in {1..20}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 1 lenet_cifar10_template_solver_450k.prototxt ${exp_id} 450k
#done


## experiment with gamma=0.01

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBclass 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_small 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done


## experiment with gamma=0.001

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_small 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done
#
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done

## experiment with Mixture of Experts gamma=0.001
## try MoE version of hier_small first. Should reach 0.617
#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hierMoE_small 1 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma0001
#done

##################################
## different CIFAR10-FULL CNN sizes
###################################
for exp_id in {1..9}
do
	./${folder}/train_script.sh cifar10_full_hier_smaller3 1 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
done
for exp_id in {1..9}
do
	./${folder}/train_script.sh cifar10_full_hier_smaller4 1 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
done
for exp_id in {1..9}
do
	./${folder}/train_script.sh cifar10_full_hier_smaller5 1 cifar10_full_template_solver.prototxt ${exp_id} 60k_gamma1_bsize500
done
