import glob, os
import json
import wandb

from .pytorch_image_models.timm.models.modified import resnet
from .pytorch_image_models.timm.models.modified import densenet
from .pytorch_image_models.timm.models.modified import convnext

import torch
import torch.nn as nn
from torch import Tensor


MODELS = {
    "resnet18": resnet.resnet18,
    # "ResNet34": resnet.resnet34,
    # "ResNet50": resnet.resnet50,
    "densenet121": densenet.densenet121,
    "seresnet18": resnet.seresnet18
}

## NOTE: for some of these (new) AFs, self.inplace = inplace is just a placeholder, although it is not actually used in the function

class new_AF1(nn.Module):
    def __init__(self, inplace=False):
        super(sinarctan, self).__init__()
        self.inplace = inplace
        
    def forward(self, input: Tensor) -> Tensor:
        output = torch.sign(input)*(torch.pow(torch.abs(input)+1, 0.33333)-1)
        return output

class sinarctan(nn.Module):
    def __init__(self, inplace=False):
        super(sinarctan, self).__init__()
        self.inplace = inplace 

    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(torch.arctan(input))

class x2_sqrtx2plus5(nn.Module):
    def __init__(self, inplace=False):
        super(x2_sqrtx2plus5, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # attempt1
        # output = input.clone()
        # output[output > 0] = input**2/torch.sqrt(input**2+5)
        
        # attempt 2
        # if input > 0:
        #     output = input**2/torch.sqrt(input**2+5)
        # else:
        #     output = -input**2/torch.sqrt(input**2+5)
        # return output

        # attempt3
        output = torch.where(input >= 0, input**2/torch.sqrt(input**2+5), -input**2/torch.sqrt(input**2+5))
        return output

class sqrtx(nn.Module):
    def __init__(self, inplace=False):
        super(sqrtx, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # print('input', input)
        # print('')
        output = torch.where(input >= 0, torch.sqrt(input+1), -torch.sqrt(torch.abs(input)+1))
        # print('output', output)
        return output


def odd_pow(input, exponent):
    return input.sign() * input.abs().pow(exponent)

class cbrtx(nn.Module):
    def __init__(self, inplace=False):
        super(cbrtx, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        output = odd_pow(input, 1/3)
        return output

class elu_modified(nn.Module):
    def __init__(self, inplace=False):
        super(elu_modified, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # print('input', input)
        # output = torch.max(0,input) + torch.min(0, (torch.exp(input)-1)) # ori
        # zero = torch.tensor(0.0)
        # output = torch.max(zero,torch.divide(input, 2)) + torch.min(zero, (torch.exp(input)-1)) # mod

        output = torch.where(input >= 0, torch.multiply(input,0.5), nn.ELU()(input))
        # print('output', output)
        return output 

class relu_modified(nn.Module):
    def __init__(self, inplace=False):
        super(relu_modified, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        output = torch.where(input >= 0, torch.multiply(input,0.5), nn.ReLU(inplace=self.inplace)(input))
        return output 

class celu_modified(nn.Module):
    def __init__(self, inplace=False):
        super(celu_modified, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        output = torch.where(input >= 0, torch.multiply(input,0.5), nn.CELU(inplace=self.inplace)(input))
        return output 


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
    "glu": nn.GLU,
    "new_AF1": new_AF1,
    "new_sinarctan": sinarctan, #inspiration: https://www.planetanalog.com/improving-a-robot-controller-replacing-tanhx-with-sinarctanx/#
    "new_x2_sqrtx2plus5": x2_sqrtx2plus5, 
    "new_sqrtx": sqrtx, #gives nan
    "new_cbrtx": cbrtx, #gives nan
    "new_elu_modified": elu_modified,
    "new_relu_modified": relu_modified,
    "new_celu_modified": celu_modified

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
