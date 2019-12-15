import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, ):
        super(UNet, self).__init__()
        self.inc= DoubleConv(5,64) 
        self.down1= DownSample(64,128)
        self.down2= DownSample(128,128)
        self.down3= DownSample(128,128)
        self.up1= UpSample(256, 128)
        self.up2= UpSample(256, 64)
        self.up3= UpSample(128, 16)
        self.outConv= OutConv(16,1)
        
    def forward(self, x):
        x1= self.inc(x) #(bsize,64,50,50)
        x2=self.down1(x1) #(bsize,128,25,25)
        x3=self.down2(x2) #(bsize,128,12,12)
        x4=self.down3(x3) #(bsize,128,6,6)

        out=self.up1(x4, x3) #(bsize,128,12,12)
        out=self.up2(out, x2)# (bsize,64,25,25)
        out=self.up3(out, x1)# (bsize,16,50,50)
        out= self.outConv(out)# (bsize,1,50,50)
        
        return out
        
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.doubleConv= nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3,1,1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True),
                        nn.Conv2d(out_channels, out_channels, 3,1,1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(True),
                )
    def forward(self, x):
        out= self.doubleConv(x)
        
        return out
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv= nn.Sequential(
                        nn.MaxPool2d(2),
                        DoubleConv(in_channels, out_channels)
                        )
        
    def forward(self, x):
        return self.maxpool_conv(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv= DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1= self.up(x1)
        diffY = x2.size()[2]-x1.size()[2]
        diffX = x2.size()[3]-x1.size()[3]
        
        x1= F.pad(x1, [diffX//2, diffX-diffX//2,
                      diffY//2, diffY-diffY//2])
        
#         print(x1.size(), x2.size())
        
        x= torch.cat([x2, x1], dim=1)

        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv= nn.Conv2d(in_channels, out_channels,3,1,1)
        
    def forward(self, x):
        return self.conv(x)