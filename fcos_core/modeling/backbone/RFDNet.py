import torch
import torch.nn as nn
import math
import torch.nn.init as init
# from .utils import load_state_dict_from_url


class DSF(nn.Module):
    def __init__(self, inp_dim, out_dim, stride=1):
        super(DSF, self).__init__()
        self.conv1    = nn.Conv2d(inp_dim, out_dim, kernel_size=1, stride=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_dim)
        self.depth_conv = nn.Conv2d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_dim
        )
        self.point_conv = nn.Conv2d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv_1x1 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=stride, bias=False)
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        conv2 = self.depth_conv(bn1)
        conv3 = self.point_conv(conv2)
        bn2   = self.bn2(conv3)
        conv_1x1   = self.conv_1x1(bn2)

        return self.relu(conv_1x1)


class RFDNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(RFDNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire1 = DSF(64, 64)
        self.fire2 = DSF(64, 64)
        self.fire3 = DSF(64, 128)
        self.fire4 = DSF(128, 128)
        self.fire5 = DSF(128, 256)
        self.fire6 = DSF(256, 256)
        self.fire7 = DSF(256, 512)
        self.fire8 = DSF(512, 512)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=8)

        self.conv256 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.conv512 = nn.Conv2d(64, 512, kernel_size=1, stride=1)
        self.conv1024 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
        self.conv2048 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)

        self.conv512d = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.conv1024d = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        self.conv2048d = nn.Conv2d(2048, 256, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.bn1024 = nn.BatchNorm2d(1024)

    def forward(self,x):
        c1 = self.conv1(x)
        c2 = self.maxpool(c1)

        f1 = self.fire1(c2)
        f2 = self.fire2(f1)

        c3 = self.maxpool(f2)

        f3 = self.fire3(c3)
        f4 = self.fire4(f3)

        c4 = self.maxpool(f4)

        f5 = self.fire5(c4)
        f6 = self.fire6(f5)
        f7 = self.fire7(f6)
        f8 = self.fire8(f7)

        m1 = self.maxpool(f2)
        m2 = self.maxpool(f4)
        m3 = self.maxpool(f8)

        c256 = self.relu(self.bn256(self.conv256(m1)))
        c512 = self.relu(self.bn512(self.conv512(m1)))
        c1024 = self.relu(self.bn512(self.conv1024(m2)))
        c2048 = self.relu(self.bn1024(self.conv2048(m3)))

        rc512 = self.maxpool2(c512)
        c1024 = torch.cat((rc512, c1024), 1)
        rc1024 = self.maxpool2(c1024)
        c2048 = torch.cat((rc1024, c2048), 1)
        
#        P256 = c256
#        P512 = self.relu(self.bn256(self.conv512d(c512)))
#        P1024 = self.relu(self.bn256(self.conv1024d(c1024)))
#        P2048 = self.relu(self.bn256(self.conv2048d(c2048)))

        return c256,c512,c1024,c2048
