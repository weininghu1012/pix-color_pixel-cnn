from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
import math


# create the conditioning network
def conditioning_network(**kwargs):
    return models.resnet101(True)

# Todo: Luminance
# Todo: Concat

class adaptation_network(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=1):
        super(adaptation_network, self).__init__()
        self.conv1 = nn.Conv2d(in_size, 512, kernel_size, bias=False)
        self.conv2 = nn.Conv2d(512, 256, kernel_size, bias=False)
        self.conv3 = nn.Conv2d(256, out_size, kernel_size, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


class PixColor1(nn.Module):

    def __init__(self, downsize):
        super(PixColor1, self).__init__()
        self.conditioning_network = conditioning_network()
        self.adaptation_network = adaptation_network(1025, 2)
        self.pixelCNN =
        '''transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Scale(downsize, interpolation=2),
            transforms.ToTensor()
            ])'''
        self.resize = transforms.Scale(downsize, interpolation=2)

    def forward(self, x):
        x_conditioned = self.conditioning_network(x)         # 64 x 1024 x 4 X 4 (4 if input imsize is 64)
        # print(x_conditioned.size())
        x_resized = self.resize(x)                           # 64 x 1 x 4 x 4
        x_concat = torch.cat((x_resized, x_conditioned), 1)  # 64 x 1025 x 4 x 4

        out = self.adaptation_network(x_concat)
        out = self.pixelCNN

        return out