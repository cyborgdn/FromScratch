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
    def __init__(self):
        super(discriminator, self).__init__()

    def forward(self, x):
        pass

# The DCGAN
class dcgan(nn.Module):
    def __init__(self):
        super(dcgan, self).__init__()

    def forward(self, x):
        pass

if __name__ == '__main__':
    pass