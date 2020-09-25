import torch
import torch.nn as nn

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=3,
                     padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = conv1x1(out_planes, out_planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # print("call the downsample")
            identity = self.downsample(x)

        # print(f"out and identity's size: {out.size()}, {identity.size()}")
        out += identity
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, stages, num_classes=1000):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, stages[0], stride=1)
        self.layer2 = self._make_layer(block, 128, stages[1], stride=2)
        self.layer3 = self._make_layer(block, 256, stages[2], stride=2)
        self.layer4 = self._make_layer(block, 512, stages[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)



    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # print(f"After first max pool: {out.size()}")

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



def ResNet18():
    return Resnet(BasicBlock, [2, 2, 2, 2])

def ResNet101():
    return Resnet(Bottleneck, [3, 4, 23, 3])
















