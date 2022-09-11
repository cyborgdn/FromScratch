from turtle import forward
import torch
import torch.nn as nn

# The generator
class generator(nn.Module):
    def __init__(self, in_features):
        super(generator, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(in_features, 1024, kernel_size=4, stride=2)
        self.dconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv5 = nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu=nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.dconv1(x)))
        x = self.relu(self.bn2(self.dconv2(x)))
        x = self.relu(self.bn3(self.dconv3(x)))
        x = self.relu(self.bn4(self.dconv4(x)))
        x = self.tanh(self.dconv5(x))
        return x

# The discriminator
class discriminator(nn.Module):
    def __init__(self, in_features):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 128, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        self.relu=nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.sig(self.conv5(x))
        return x

# The DCGAN
class dcgan(nn.Module):
    def __init__(self, in_features, img_channels):
        super(dcgan, self).__init__()
        self.generator = generator(in_features=in_features)
        self.discriminator = discriminator(in_features=img_channels)

    def forward(self, x):
        x = self.discriminator(x)
        # assert x.shape == (N, 1, 1, 1), "Discriminator test failed"
        x = self.generator(x)
        # assert x.shape == (N, in_channels, H, W), "Generator test failed"
        return x