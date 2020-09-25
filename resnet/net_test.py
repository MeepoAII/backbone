import torch
import torchvision
from resnet.resnet_v1 import *
net = torchvision.models.resnet101(pretrained=False)
# net = ResNet101()
print(net)
x = torch.randn(4, 3, 224, 224)
out = net(x)
print(out.size())
# print(net)
