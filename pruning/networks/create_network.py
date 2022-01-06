import pruning.networks.masked_vgg as masked_vgg
import pruning.networks.masked_vgg_64 as masked_vgg_64
import pruning.networks.masked_resnet as masked_resnet
import pruning.networks.masked_resnet_64 as masked_resnet_64

def get_network(network, *args, **kwargs):
    mapping = {
            'vgg11': masked_vgg.MaskedVGG11,
            'vgg13': masked_vgg.MaskedVGG13,
            'vgg16': masked_vgg.MaskedVGG16,
            'vgg19': masked_vgg.MaskedVGG19,
            'vgg11_64': masked_vgg_64.MaskedVGG11_64,
            'vgg13_64': masked_vgg_64.MaskedVGG13_64,
            'vgg16_64': masked_vgg_64.MaskedVGG16_64,
            'vgg19_64': masked_vgg_64.MaskedVGG19_64,
            'resnet18': masked_resnet.MaskedResNet18,
            'resnet34': masked_resnet.MaskedResNet34,
            'resnet50': masked_resnet.MaskedResNet50,
            'resnet101': masked_resnet.MaskedResNet101,
            'resnet152': masked_resnet.MaskedResNet152,
            'resnet18_64': masked_resnet_64.MaskedResNet18_64,
            'resnet34_64': masked_resnet_64.MaskedResNet34_64,
            'resnet50_64': masked_resnet_64.MaskedResNet50_64,
            'resnet101_64': masked_resnet_64.MaskedResNet101_64,
            'resnet152_64': masked_resnet_64.MaskedResNet152_64,
    }
    return mapping[network.lower()](*args, **kwargs)
