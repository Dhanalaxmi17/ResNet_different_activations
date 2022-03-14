import glob, os
import json
import wandb

from .pytorch_image_models.timm.models.modified import resnet
from .pytorch_image_models.timm.models.modified import densenet
from .pytorch_image_models.timm.models.modified import convnext

import torch
import torch.nn as nn


MODELS = {
    "resnet18": resnet.resnet18,
    # "ResNet34": resnet.resnet34,
    # "ResNet50": resnet.resnet50,
    "densenet121": densenet.densenet121,
    "seresnet18": resnet.seresnet18
}

ACTIVATION_FUNCTION = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "hardswish": nn.Hardswish,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "glu": nn.GLU
}

def create_model(model_str, af_str, **kwargs):
    try:
        return MODELS[model_str](act_layer = ACTIVATION_FUNCTION[af_str], **kwargs)    
    except:
        return print("Model name not recognized or model not registered.")
    
    # elif model_str == "DenseNet121":
    #     wandb.config.update({"Model": model_str})
    #     return densenet.densenet121(act_layer= ACTIVATION_FUNCTION[af_str], **kwargs)
    
    # elif model_str == "SEResNet18":
    #     wandb.config.update({"Model": model_str})
    #     return resnet.seresnet18(act_layer= ACTIVATION_FUNCTION[af_str], **kwargs)
    
    # else:
    #     return print("Model name not recognized or model not registered.")