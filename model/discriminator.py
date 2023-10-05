import torch.nn as nn
import functools
import torch
from pytorch_model_summary import summary


class PixelDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, num_filters=(128, 256, 512, 512)):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_filters (int)       -- the number of filters in the conv layer
        """
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_nc, num_filters[0], kernel_size=4, padding=1, stride=2, bias=True),
                                   nn.LeakyReLU(0.2, True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(num_filters[0], num_filters[1], kernel_size=4, padding=1, stride=2, bias=True),
                                   nn.LeakyReLU(0.2, True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(num_filters[1], num_filters[2], kernel_size=4, padding=1, stride=2, bias=True),
                                   nn.LeakyReLU(0.2, True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(num_filters[2], num_filters[3], kernel_size=4, padding=1, stride=1, bias=True),
                                   nn.LeakyReLU(0.2, True))
        
        self.out_conv = nn.Sequential(nn.Conv2d(num_filters[3], 1, kernel_size=4, padding=1, stride=1, bias=True))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out_conv(x)
        out = torch.sigmoid(x)
        return out
    

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    x = torch.ones([4, 3, 256, 256]).cuda()
    model = PixelDiscriminator(3).cuda()

    print(summary(model,x))
    print('input:',x.shape)
    print('output:',model(x).shape)
    print('===================================')
    print(model.parameters)