import torch.nn as nn
import torch
import math
import constants

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetCombine(nn.Module):

    def __init__(self, block, layers, num_classes=3, mode="full"):
        self.inplanes_laser = 64
        self.num_classes = num_classes
        self.inplanes_map = 64
        super(ResNetCombine, self).__init__()
        self.conv1_laser = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_laser = nn.BatchNorm2d(64)
        self.relu_laser = nn.ReLU(inplace=True)
        self.maxpool_laser = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_laser = self._make_layer(block, 64, layers[0], is_map=False)
        self.layer2_laser = self._make_layer(block, 128, layers[1], stride=2, is_map=False)
        # self.layer3_laser = self._make_layer(block, 256, layers[2], stride=2, is_map=False)
        self.mode = mode
        self.conv1_map = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_map = nn.BatchNorm2d(64)
        self.relu_map = nn.ReLU(inplace=True)
        self.maxpool_map = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_map = self._make_layer(block, 64, layers[0])
        self.layer2_map = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3_map = self._make_layer(block, 256, layers[2], stride=2)
        if mode=="full":
            self.inplanes_map += 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(256 * 4, constants.GRID_LENGTH*constants.GRID_LENGTH*1*(3+num_classes))
        self.conv_last =  nn.Conv2d(256, 3+num_classes, kernel_size=4, stride=1, bias=False)
        self.drop_out = nn.Dropout()
        self.soft_max = nn.LogSoftmax(dim=4)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, is_map=True):
        downsample = None
        inplanes = self.inplanes_laser
        if is_map:
            inplanes = self.inplanes_map

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        if is_map:
            self.inplanes_map = inplanes
        else:
            self.inplanes_laser = inplanes
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, laser, map):
        laser = self.conv1_laser(laser)
        laser = self.bn1_laser(laser)
        laser = self.relu_laser(laser)
        laser = self.maxpool_laser(laser)

        laser = self.layer1_laser(laser)
        laser = self.layer2_laser(laser)
        # laser = self.layer3_laser(laser)

        map = self.conv1_map(map)
        map = self.bn1_map(map)
        map = self.relu_map(map)
        map = self.maxpool_map(map)

        map = self.layer1_map(map)
        map = self.layer2_map(map)
        # map = self.layer3_map(map)
        concat = torch.cat((map, laser), 1)
        concat = self.drop_out(concat)
        if self.mode == "laser":
            concat = laser
        elif self.mode == "map":
            concat = map
        x = self.layer3(concat)
        x = self.layer4(x)

        x = self.conv_last(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        predict = x.view(x.size(0), constants.GRID_LENGTH, constants.GRID_LENGTH, 1, (3+self.num_classes))

        classes_out = self.soft_max(predict[:, :, :, :, 3:])
        pose_objectness = predict[:, :, :, :, 0:3]  # (self.tanh(predict[:,:,:,:,0:3]) + 1.0) / 2
        pose_objectness = (self.tanh(pose_objectness) + 1.0) / 2
        poses = pose_objectness[:, :, :, :, 1:3]
        objectness = pose_objectness[:, :, :, :, 0]
        # return poses_out, classes_out
        return classes_out, poses, objectness

        return x


def my_resnet(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetCombine(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
