def get_default_hparams(network_type):
    if network_type == "vgg11":
        return {
                "use_bn": True,
                "prune_ratios": [.15]*8 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iters": 35000,
                "finetune_iters": 25000,
                "batch_size": 60
        }
    elif network_type == "vgg16":
        return {
                "use_bn": True,
                "prune_ratios": [.15]*13 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iters": 50000,
                "finetune_iters": 35000,
                "batch_size": 60
        }
    elif network_type == "vgg19":
        return {
                "use_bn": True,
                "prune_ratios": [.15]*16 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iters": 60000,
                "finetune_iters": 40000,
                "batch_size": 60
        }
    elif network_type == 'vgg19_64':
        return {
                "use_bn": True,
                "prune_ratios": [.15] * 16 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iters": 60000,
                "finetune_iters": 40000,
                "batch_size": 60
        }
    elif network_type == 'resnet18':
        return {
                "prune_ratios": [0] + [.15] * 16 + [.10],
                "optimizer": 'adam',
                "lr": 0.0003,
                "pretrain_iters": 35000,
                "finetune_iters": 25000,
                "batch_size": 60
        }
    elif network_type == 'resnet50_64':
        return {
                "prune_ratios": [0] + [.15] * 48 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iteration": 60000,
                "finetune_iteration": 40000,
                "batch_size": 60
        }
    elif network_type == 'resnet50':
        return {
                "prune_ratios": [0] + [.15] * 48 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iters": 60000,
                "finetune_iters": 40000,
                "batch_size": 60
        }
    elif network_type == 'resnet101':
        return {
                "prune_ratios": [0] + [.15] * 99 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iters": 80000,
                "finetune_iters": 40000,
                "batch_size": 60
        }
    elif network_type == 'resnet152':
        return {
                "prune_ratios": [0] + [.15] * 150 + [.10],
                "optimizer": "adam",
                "lr": 0.0003,
                "pretrain_iters": 100000,
                "finetune_iters": 40000,
                "batch_size": 60
        }
    elif network_type == 'resnet64':
	    return {
            "prune_ratios" : [0] + [.15] * 12 + [.10],
            "optimizer" : "adam",
            "pretrain_iters" : 60000,
            "finetune_iters" : 40000,
            "batch_size" : 60
        }
    else:
        print('No default hparams found')
        return {}
