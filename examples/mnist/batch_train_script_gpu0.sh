#!/bin/bash


folder='examples/mnist'
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier 0 lenet_template_solver.prototxt ${exp_id} gamma001
#done
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier_small 0 lenet_template_solver.prototxt ${exp_id} gamma001
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier 0 lenet_template_solver.prototxt ${exp_id} gamma01
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_hier 0 lenet_template_solver.prototxt ${exp_id} gamma1
#done

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_smallBranch 0 lenet_template_solver.prototxt ${exp_id} origin
#done

for exp_id in {1..9}
do
	./${folder}/train_script.sh lenet_small 0 lenet_template_solver.prototxt ${exp_id} origin
done
####################################################
##### CIFAR 10 experiments
####################################################

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

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001
#done

#for exp_id in {1..12}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier3_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma01
#done

#####################################################
### see how base lr affects accuracy
#####################################################

#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hierMoE_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr01
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hier_smallBranch 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done
#
#for exp_id in {1..9}
#do
#	./${folder}/train_script.sh lenet_cifar10_hierMoE_small 0 lenet_cifar10_template_solver.prototxt ${exp_id} 100k_gamma001_lr0001
#done



