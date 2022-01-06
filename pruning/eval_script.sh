#!/bin/bash

# Color
BLUE='\033[0;34m'
NC='\033[0m'

ID=$1
NETWORK=$2
LOADDIR=$3
echo "ID: $ID"
echo "Network: $NETWORK"

# Don't use gpu:2
#if [[ $ID = 2 ]]
#then
#    exit
#fi

# Finetune iters
if [[ $NETWORK = 'vgg11' ]]
then
    FINETUNE=25000
elif [[ $NETWORK = 'vgg16' ]]
then
    FINETUNE=35000
elif [[ $NETWORK = 'vgg19' ]]
then
    FINETUNE=40000
elif [[ $NETWORK = 'vgg19_64' ]]
then
    FINETUNE=40000
elif [[ $NETWORK = 'resnet18' ]]
then
    FINETUNE=25000
elif [[ $NETWORK = 'resnet50_64' ]]
then
    FINETUNE=40000
fi

# Set load iters (TODO: automate this)
if [[ $ID = 0 ]]
then
    iters=(8192 16384 24576 32768 40960)
elif [[ $ID = 1 ]]
then
    iters=(49152 57344 65536 73728 81920)
elif [[ $ID = 2 ]]
then
    #iters=(90112 98304 106496 114688 122880)
    #iters=(104448 110592 116736 122880)
    iters=(104448 122880)
fi

# Now run with and without thresholding
for i in "${iters[@]}"
do
    echo -e -n "$BLUE"
    echo -e "=================================="
    echo -e "Iter $i (Without thresholding)"
    echo -e "=================================="
    echo -e -n "$NC"
    CUDA_VISIBLE_DEVICES=$ID python -m pruning.run_policy -e FP -efi $FINETUNE -eutd -li $i -en $NETWORK -l $LOADDIR
    echo -e -n "$BLUE"
    echo -e "=================================="
    echo -e "Iter $i (With thresholding)"
    echo -e "=================================="
    echo -e -n "$NC"
    CUDA_VISIBLE_DEVICES=$ID python -m pruning.run_policy -e FP -efi $FINETUNE -eutd -li $i -en $NETWORK -eha -l $LOADDIR
done
