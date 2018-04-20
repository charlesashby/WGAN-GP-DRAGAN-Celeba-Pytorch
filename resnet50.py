from torch.autograd import Variable
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        hidden_state = []

        x = self.layer1(x)
        hidden_state.append(x.clone())
        x = self.layer2(x)
        hidden_state.append(x.clone())
        x = self.layer3(x)
        hidden_state.append(x.clone())
        x = self.layer4(x)
        hidden_state.append(x.clone())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, hidden_state

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))

    model.fc = nn.Linear(512 * 4, 2)
    return model

def calculate_perceptual_distance(y0, y1, w=1):
    # hidden's structure: (batch, channel, h, w)
    diff = w*(y0 - y1)
    norm = torch.norm(diff, p=2, dim=1)
    sum1 = torch.sum(norm, dim=2)
    sum2 = torch.sum(sum1, dim=1)
    distance = sum2/(y0.size(2)*y0.size(3))

    return distance

def perceptual_distance(model, x0, x1):
    distance = 0
    _, y0 = model(x0)
    _, y1 = model(x1)

    for l in range(len(y0)):
        distance += calculate_perceptual_distance(y0[l], y1[l])
    return distance

if __name__=='__main__':
    print("Loading the model...")
    resnet = resnet50()
    resnet.cuda()
    print("Get example")
    n_batch = 32
    example1 = torch.Tensor(n_batch, 3, 224, 224).uniform_(-1, 1).cuda()
    example1 = Variable(example1)
    example2 = torch.Tensor(n_batch, 3, 224, 224).uniform_(-1, 1).cuda()
    example2 = Variable(example2)
    perceptual_distance(resnet, example1, example2)

