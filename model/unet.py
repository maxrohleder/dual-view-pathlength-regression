import torch
import torch.nn as nn
from fume import Fume3dLayer


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                                stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)

        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)

        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x


class DualViewUNet(nn.Module):
    def __init__(self, out_classes=1, up_sample_mode='conv_transpose'):
        super(DualViewUNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        self.fume = Fume3dLayer()

        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)

        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)

        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, P: torch.Tensor):
        """
        jointly segment two views of 3d scene
        :param x: expected shape is (b, 2, 976, 976)
        :param P: expected shape is (b, 2, 3, 3)
        :return: tensor in shape (b, 2, 976, 976)
        """

        # split views stored in input image channels
        view1 = x.select(1, 0).unsqueeze(1)  # (dim, index)
        view2 = x.select(1, 1).unsqueeze(1)

        # select respective fundamental matrices P = [F12, F21]
        F12 = P.select(1, 0).contiguous()
        F21 = P.select(1, 1).contiguous()

        # down block level 1
        view1, view1_skip1 = self.down_conv1(view1)
        CM2 = self.fume(view1, F21, F12)
        view2, view2_skip1 = self.down_conv1(view2)
        CM1 = self.fume(view2, F12, F21)
        view1 = torch.cat([view1, CM1], dim=1)
        view2 = torch.cat([view2, CM2], dim=1)

        # down block level 2
        view1, view1_skip2 = self.down_conv2(view1)
        CM2 = self.fume(view1, F21, F12)
        view2, view2_skip2 = self.down_conv2(view2)
        CM1 = self.fume(view2, F12, F21)
        view1 = torch.cat([view1, CM1], dim=1)
        view2 = torch.cat([view2, CM2], dim=1)

        # down block level 3
        view1, view1_skip3 = self.down_conv3(view1)
        CM2 = self.fume(view1, F21, F12)
        view2, view2_skip3 = self.down_conv3(view2)
        CM1 = self.fume(view2, F12, F21)
        view1 = torch.cat([view1, CM1], dim=1)
        view2 = torch.cat([view2, CM2], dim=1)

        # down block level 4
        view1, view1_skip4 = self.down_conv4(view1)
        view2, view2_skip4 = self.down_conv4(view2)
        CM2 = self.fume(view1, F21, F12)
        CM1 = self.fume(view2, F12, F21)
        view1 = torch.cat([view1, CM1], dim=1)
        view2 = torch.cat([view2, CM2], dim=1)

        # bottleneck
        view1 = self.double_conv(view1)
        view2 = self.double_conv(view2)

        # up block level 4
        CM2_skip4 = self.fume(view1_skip4, F21, F12)
        CM1_skip4 = self.fume(view2_skip4, F12, F21)
        view1_skip4 = torch.cat([view1_skip4, CM1_skip4], dim=1)
        view2_skip4 = torch.cat([view2_skip4, CM2_skip4], dim=1)
        view1 = self.up_conv4(view1, view1_skip4)
        view2 = self.up_conv4(view2, view2_skip4)

        # up block level 3
        CM2_skip3 = self.fume(view1_skip3, F21, F12)
        CM1_skip3 = self.fume(view2_skip3, F12, F21)
        view1_skip3 = torch.cat([view1_skip3, CM1_skip3], dim=1)
        view2_skip3 = torch.cat([view2_skip3, CM2_skip3], dim=1)
        view1 = self.up_conv4(view1, view1_skip3)
        view2 = self.up_conv4(view2, view2_skip3)

        # up block level 2
        CM2_skip2 = self.fume(view1_skip2, F21, F12)
        CM1_skip2 = self.fume(view2_skip2, F12, F21)
        view1_skip2 = torch.cat([view1_skip2, CM1_skip2])
        view2_skip2 = torch.cat([view2_skip2, CM2_skip2])
        view1 = self.up_conv4(view1, view1_skip2)
        view2 = self.up_conv4(view2, view2_skip2)

        # up block level 1
        CM2_skip1 = self.fume(view1_skip1, F21, F12)
        CM1_skip1 = self.fume(view2_skip1, F12, F21)
        view1_skip1 = torch.cat([view1_skip1, CM1_skip1])
        view2_skip1 = torch.cat([view2_skip1, CM2_skip1])
        view1 = self.up_conv4(view1, view1_skip1)
        view2 = self.up_conv4(view2, view2_skip1)

        # last conv
        view1 = self.conv_last(view1)
        view2 = self.conv_last(view2)

        # recombine views into channels
        x = torch.cat([view1, view2], dim=1)
        return x
