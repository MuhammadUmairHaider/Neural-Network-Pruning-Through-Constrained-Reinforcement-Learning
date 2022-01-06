#!/bin/bash

# VGG11
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en vgg11 --group fsp-vgg11 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b 20 -tefi 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000 -nt 1
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en vgg11 --group fsp-vgg11 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b 10 -tefi 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en vgg11 --group fsp-vgg11 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b  5 -tefi 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en vgg11 --group fsp-vgg11 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b  2 -tefi 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en vgg11 --group fsp-vgg11 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b  1 -tefi 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000



#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en vgg11 --group fsp-vgg11 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b 20 -tefi 0 256 512 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000

#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en resnet18 --group fsp-resnet18 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b 20 -tefi 0 0 32 64 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000 -nt 3
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en resnet18 --group fsp-resnet18 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b 10 -tefi 0 0 32 64 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000 -nt 3
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg -en resnet18 --group fsp-resnet18 -tei FP -eei FP -ns 128 -piv 1 -plr 1 -pmv 0. -b 1  -tefi 0 0 32 64 256 -ebs 20000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 124000 -nt 3



#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg --group cg-vgg11 -en vgg11 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -b 20 -ebs 20000 -eefi 35000 -ee -1 -teaci 1. -teacg 0. -cgl 1. -t 40000 -nt 3


## VGG11
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg --group cg-vgg11 -en vgg11 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -b 20 -ebs 20000 -ee -1 -teaci 0.8 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg --group cg-vgg11 -en vgg11 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -b 10 -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
#
## VGG 16
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg --group cg-vgg16 -en vgg16 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -b 20 -ebs 20000 -ee -1 -teaci 0.8 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg --group cg-vgg16 -en vgg16 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -b 10 -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
#
## VGG 19
#echo "HEY"
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg --group cg-vgg19 -en vgg19 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -b 20 -ebs 20000 -ee -1 -teaci 0.8 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
#CUDA_VISIBLE_DEVICES=2 python -m pruning.cpg --group cg-vgg19 -en vgg19 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -b 10 -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128


#python -m cpg.run_policy -l Pruning/32wbawov -r -li 122880 -e FP -efi 40000 -eutd -en vgg19
#echo "vgg19 a=10"
#python -m pruning.run_policy -l Pruning/ns1q2oua -r -li 122880 -e FP -efi 40000 -eutd -en vgg19

#echo "vgg16 a=20"
#CUDA_VISIBLE_DEVICES=2 python -m pruning.run_policy -l Pruning/fz2ysv15 -r -li 122880 -e FP -efi 35000 -eutd -en vgg16 -eha
#echo "vgg16 a=10"
#CUDA_VISIBLE_DEVICES=2 python -m pruning.run_policy -l Pruning/33u58bx6 -r -li 122880 -e FP -efi 35000 -eutd -en vgg16 -eha
#
#echo "vgg11 a=20"
#CUDA_VISIBLE_DEVICES=2 python -m pruning.run_policy -l Pruning/2mhj5icm -r -li 122880 -e FP -efi 25000 -eutd -en vgg11 -eha
#echo "vgg11 a=10"
#CUDA_VISIBLE_DEVICES=2 python -m pruning.run_policy -l Pruning/2p5buz4s -r -li 122880 -e FP -efi 25000 -eutd -en vgg11 -eha

#CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 9-13 python -m pruning.train --network vgg11 -pm MP -pr 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 -pi 1
#CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 9-13 python -m pruning.train --network vgg11 -pm MP -pr 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 -pi 1

CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 9-13 python -m pruning.train --network vgg16 -pm MP -pr 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 -pi 1
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 9-13 python -m pruning.train --network vgg16 -pm MP -pr 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 -pi 1

CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 9-13 python -m pruning.train --network vgg19 -pm MP -pr 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 -pi 1
CUDA_VISIBLE_DEVICES=2 taskset --cpu-list 9-13 python -m pruning.train --network vgg19 -pm MP -pr 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 -pi 1
