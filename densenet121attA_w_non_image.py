        # Code snippet for freezing layer weights
        #for param in model.named_parameters():
        #    if  param[0].startswith("features.denseblock1") or \
        #        param[0].startswith("features.transition1") or \
        #        param[0].startswith("features.denseblock2") or \
        #        param[0].startswith("features.transition2") or \
        #        param[0].startswith("features.denseblock3") or \
        #        param[0].startswith("features.transition3") or \
        #        param[0].startswith("features.denseblock4") or \
        #        param[0].startswith("features.norm5") or \
        #        param[0].startswith("classifier"): 
        #        # make this layer trainable
        #        pass
        #    else:
        #        # freeze this layer
        #        param[1].requires_grad = False
'''
Densenet
'''
# !!! MODIFIED FROM: Original PyTorch Implementation of DenseNet (torchvision 0.9)
# https://github.com/pytorch/vision/blob/release/0.9/torchvision/models/densenet.py

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple

import torch.nn.init as nn_init

# Available DenseNet models
#__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient


    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
            
        # !!! ADDED BY USER !!!   
        # attention estimators
        self.att_est1 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.att_est2 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.att_est3 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        
        # projection layers, note this is DIFFERENT from the original paper.
        # instead of projecting g to l, we will project l to g, since l1, l2, l3 are all smaller then g
        # reference:
        # https://towardsdatascience.com/learn-to-pay-attention-trainable-visual-attention-in-cnns-87e2869f89f1 - "What if l and g are not the same size?"
        self.att_proj1 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.att_proj2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.att_proj3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Non-image model
        n_non_im_features = 3
        self.non_image_fc = nn.Linear(n_non_im_features, n_non_im_features)
        
        # Linear layer
        self.classifier = nn.Linear((1024*3) + n_non_im_features, num_classes)
    
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    # !!! ADDED BY USER !!! #
    def calculate_ga(self, l, g, att_est):
        num_batch, num_channel, width, height = l.shape
        
        # "Learn to Pay Attention" Eq.2
        # https://towardsdatascience.com/learn-to-pay-attention-trainable-visual-attention-in-cnns-87e2869f89f1 - Step 1
        # Note: c should be per pixel, NOT per channel, see Notation part of the link above.
        c = att_est(l + g)
        
        # "Learn to Pay Attention" Eq.1
        # https://towardsdatascience.com/learn-to-pay-attention-trainable-visual-attention-in-cnns-87e2869f89f1 - Step 2
        # Note: we need to perform softmax on all the pixels, so first we need to flatten dim=-2 and dim=-1 of c, 
        #       after softmax we need to convert them back to original shape.
        img_shape = c.shape
        a = c.view(num_batch, 1, img_shape[2] * img_shape[3]) # flatten dim=-2 and dim=-1
        a = F.softmax(a, dim=2)
        a = a.view(num_batch, 1, img_shape[2], img_shape[3]) # reshape to original shape
        
        # https://towardsdatascience.com/learn-to-pay-attention-trainable-visual-attention-in-cnns-87e2869f89f1 - Step 3
        a = a.expand(a.shape[0], num_channel, a.shape[2], a.shape[3])
        # a.shape = (num_batch, num_channels, a.shape[2], a.shape[3])
        ga = torch.mul(a, l); # element-wise multiplication
        # ga.shape = a.shape = l.shape
        
        ga = ga.view(ga.shape[0], ga.shape[1], ga.shape[2] * ga.shape[3])
        
        ga = ga.sum(dim=2)
        # ga.shape = (num_batch, num_channels)

        return ga
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # !!! MODIFIED BY USER !!!
#       features = self.features(x)
        
        # Initial layers
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        
        # denseblock1
        x = self.features.denseblock1(x)
        
        # transition1
        x = self.features.transition1(x)
        l1 = x
 
        # denseblock2
        x = self.features.denseblock2(x)
        
        # transition2
        x = self.features.transition2(x)
        l2 = x
        
        # denseblock3
        x = self.features.denseblock3(x)
        
        # transition3
        x = self.features.transition3(x)
        l3 = x
        
        # denseblock4
        x = self.features.denseblock4(x)
        
        # norm5
        features = self.features.norm5(x)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        g = out
        
        l1 = self.att_proj1(l1)
        l2 = self.att_proj2(l2)
        l3 = self.att_proj3(l3)
                                  
        ga1 = self.calculate_ga(l1, g, self.att_est1)
        ga2 = self.calculate_ga(l2, g, self.att_est2)
        ga3 = self.calculate_ga(l3, g, self.att_est3)
        
        # Classifier layer has been replaced
        #out = torch.flatten(out, 1)
        #out = self.classifier(out)
        
        # Non-image model
        non_image_out = F.relu(self.non_image_fc(y))
        
        #print(ga1.shape, ga2.shape, ga3.shape, g.shape)
        g_all = torch.cat((ga1, ga2, ga3, non_image_out), dim=1)
        
        out = self.classifier(g_all)
        
        return out
    
# !!! MODIFIED BY USER
def _load_state_dict_att(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    
    # ADDED BY USER
    #print('State dict = ', state_dict.keys())
    #model_param_names = [param[0] for param in  model.named_parameters()]
    
    # Add model parameters that are not in the loaded state_dict to state_dict
    for param in model.named_parameters():
        if param[0] not in state_dict.keys():
             state_dict[param[0]] = param[1]
    # Classifier exists in both models but they have different sizes
    model_classifier = model.classifier
    for m in model_classifier.modules():
        if isinstance(m, nn.Linear):
            state_dict['classifier.weight'] = m.weight
            if m.bias is not None:
                state_dict['classifier.bias'] = m.bias
            
    #print('State dict = ', state_dict.keys())
    model.load_state_dict(state_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if pretrained:
        # model_urls MOVED BY THE USER FROM OUTSIDE THE FUNCTION TO INSIDE
        model_urls = {
            'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
            'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
            'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
            'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
        }
        _load_state_dict_att(model, model_urls[arch], progress)
    return model


def densenet121attA_w_non_image(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


