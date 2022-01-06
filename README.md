### Code Overview
This code contains two folder:
    - `stable_baselines3`: This is a fork of [this](https://github.com/DLR-RM/stable-baselines3) repo, but has been substantially modified so as to add the PPO-Lagrangian algorithm that we am using.
    - `pruning`: This contains code to construct the RL environment and train an agent using PPO-Lagrangian.

The main file to run is `pruning/cpg.py`. All hyperparameters can be passed through the command line (see the argparse setup from line 148).

### Setup
To install the required pacakges, run:```shell pip install -r requirements.txt```. Optionally, also setup a [W&B account](https://wandb.ai/) to visualize metrics in real-time.

### Pretrained models
Due to shortage of space, pretrained models are provided using the following [link](https://drive.google.com/drive/folders/1UgS9UTuWUwtGoQYpxii4NVFWztTPbrtA?usp=sharing). Copy paste the pretrained models in pruning/pretrained folder.

### Experiments
To specify a different budget, modify the value after ```-b```.

#### Coarse-grained
VGG11:
```shell
python -m pruning.cpg --group cg-vgg11 -en vgg11 -b 10 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
```

VGG16:
```shell
python -m pruning.cpg --group cg-vgg16 -en vgg16 -b 10 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
```

VGG19:
```shell
python -m pruning.cpg --group cg-vgg19 -en vgg19 -b 10 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 3 -tefi 0 32 128
```

ResNet18:
```shell
python -m pruning.cpg --group cg-resnet18 -en resnet18 -b 10 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 0. -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 2 -tefi 0 32 128
```

ResNet50:
```shell
python -m pruning.cpg --group cg-resnet50 -en resnet50 -b 55 -tk 0.1 -ns 128 -piv 1 -plr 1 -pmv 1 -ebs 20000 -ee -1 -teaci 0.9 -teacg 0.05 -cgl 1. -t 40000 -nt 2 -tefi 0 32 128
```

Running the above commands will save the model weights at different iterations. To evaluate, run:
```shell
python -m pruning.run_policy -efi FINETUNE_ITRS -eutd -li SAVE_ITR -en NETWORK -l LOADDIR -eha
```
where `FINETUNE_ITRS` should be replaced by the number of iterations for which to finetune the network, and `SAVE_ITR` should be replaced by the iteration from which to replace the weights from. Finally, `NETWORK` should be replaced by the name of the network that we are pruning (i.e., 'vgg11', 'vgg16' or 'vgg19') and `LOADIR` should be replaced by directory at which the trained weights are stored in (also printed at the start of training)


### Baselines
For magnitude-based pruning baselines, run the following:

VGG11:
```shell
python -m pruning.train --network vgg11 -pm MP -pr 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 -pi 1
```

VGG16:
```shell
python -m pruning.train --network vgg16 -pm MP -pr 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 -pi 1
```

VGG19:
```shell
python -m pruning.train --network vgg19 -pm MP -pr 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 -pi 1
```

Here 0.8 indicates that we are interested in removing 80% of the neurons from each layer

### Plots
Plots can be constructed using the pruning/plot.py file. However, this would require that all runs are synced to W&B.
