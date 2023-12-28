import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, input_channel=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        ####### change from 3 to 2 #######
        self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

# resunet
class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x

class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x

class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')
        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResUNet34(nn.Module):
    def __init__(self, fixed_feature=False, **kwargs):
        super().__init__()
        # load weight of pre-trained resnet
        self.resnet = resnet34(**kwargs)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # up conv
        l = [64, 64, 128, 256, 512]
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)
        # final conv
        self.seg = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x) # 64 125 125
        x = self.resnet.bn1(x)
        x = s1 = self.resnet.relu(x) 
        x = self.resnet.maxpool(x) # 64 63 63
        x = s2 = self.resnet.layer1(x) # 64 63 63 
        x = s3 = self.resnet.layer2(x) # 128 32 32 
        x = s4 = self.resnet.layer3(x) # 256 16 16
        x = self.resnet.layer4(x) #512 8 8

        x1 = self.u5(x, s4)
        x1 = self.u6(x1, s3)
        x1 = self.u7(x1, s2)
        x1 = self.u8(x1, s1)
        x1 = self.seg(x1)
        return x1

class PostProcessingModule(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.seg_conv1 = dilated_conv(in_channel=in_channel, out_channel=in_channel*2, dropout_rate=0.1, dilation=dilation)
        self.seg_conv2 = dilated_conv(in_channel=in_channel*2, out_channel=in_channel*4, dropout_rate=0.1, dilation=dilation)

        self.edge_conv1 = dilated_conv(in_channel=in_channel, out_channel=in_channel*2, dropout_rate=0.1, dilation=dilation)
        self.edge_conv2 = dilated_conv(in_channel=in_channel*2, out_channel=in_channel*4, dropout_rate=0.1, dilation=dilation)

        self.merge_conv1 = dilated_conv(in_channel=in_channel*8, out_channel=in_channel*4, dropout_rate=0.1, dilation=dilation)
        self.merge_conv2 = dilated_conv(in_channel=in_channel*4, out_channel=out_channel, dropout_rate=0.1, dilation=dilation)

    def forward(self, seg, edge):
        seg_1 = self.seg_conv1(seg)
        seg_2 = self.seg_conv2(seg_1)

        edge_1 = self.edge_conv1(edge)
        edge_2 = self.edge_conv2(edge_1)

        merge = torch.cat([seg_2, edge_2], dim=1)

        merge_1 = self.merge_conv1(merge)
        merge_2 = self.merge_conv2(merge_1)

        return merge_2


class AffResUNet34(nn.Module):
    def __init__(self, pretrained=False, post=False):
        super().__init__()
        # load weight of pre-trained resnet
        self.resnet = resnet34(pretrained=pretrained)

        # up conv
        l = [64, 64, 128, 256, 512]

        self.u5_seg = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6_seg = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7_seg = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8_seg = ConvUpBlock(l[1], l[0], dropout_rate=0.1)

        self.u5_edge = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6_edge = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7_edge = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8_edge = ConvUpBlock(l[1], l[0], dropout_rate=0.1)
        # final conv
        self.seg = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        self.edge = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

        if post:
            self.post = PostProcessingModule(1, 1, 0.1)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = s1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = s2 = self.resnet.layer1(x)
        x = s3 = self.resnet.layer2(x)
        x = s4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x1_seg = self.u5_seg(x, s4)
        x1_seg = self.u6_seg(x1_seg, s3)
        x1_seg = self.u7_seg(x1_seg, s2)
        x1_seg = self.u8_seg(x1_seg, s1)
        seg = self.seg(x1_seg)

        x1_edge = self.u5_edge(x, s4)
        x1_edge = self.u6_edge(x1_edge, s3)
        x1_edge = self.u7_edge(x1_edge, s2)
        x1_edge = self.u8_edge(x1_edge, s1)
        edge = self.edge(x1_edge)

        return seg, edge

    def post_process(self, seg, edge):
        return self.post(seg, edge)

class AffinityLoss(AffResUNet34):
    path_indices_prefix = "path_indices"

    def __init__(self, path_index, **kwargs):
        super(AffinityLoss, self).__init__(**kwargs)

        self.path_index = path_index

        self.n_path_lengths = len(path_index.path_indices)
        for i, pi in enumerate(path_index.path_indices):
            self.register_buffer(AffinityLoss.path_indices_prefix + str(i), torch.from_numpy(pi))

    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)

        for i in range(self.n_path_lengths): #path of length 1,2,3...
            ind = self._buffers[AffinityLoss.path_indices_prefix + str(i)] # shape: (# of paths, path_length, source)
            ind_flat = ind.view(-1) # 2*2*13090
            
            dist = torch.index_select(edge, dim=-1, index=ind_flat) # 8 52360
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2)) # 8 2 2 13090
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2) # 8 2 13090
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1) # 8 152 13090

        return aff_cat

    def forward(self, x):
        seg, edge = super().forward(x) # 56 56
        return seg, edge

    def to_loss(self, edge_out):
        aff = self.to_affinity(edge_out)

        pos_aff_loss = (-1) * torch.log(aff + 1e-5)
        neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

        return pos_aff_loss, neg_aff_loss