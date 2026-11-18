"""Native T=9 3-D U-Net; pooling/padding are spatial only."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .landslide_common import explicit_sequence,assert_output

class Block(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv3d(cin,cout,3,padding=1,bias=False),nn.BatchNorm3d(cout),nn.ReLU(inplace=True),nn.Conv3d(cout,cout,3,padding=1,bias=False),nn.BatchNorm3d(cout),nn.ReLU(inplace=True))

class LandslideUNet3D(nn.Module):
    spatial_padding=(8,8)
    def __init__(self,**_):
        super().__init__(); widths=(32,64,128,256)
        self.enc=nn.ModuleList([Block(6,widths[0]),Block(widths[0],widths[1]),Block(widths[1],widths[2]),Block(widths[2],widths[3])]); self.pool=nn.MaxPool3d((1,2,2)); self.center=Block(256,512)
        self.up=nn.ModuleList([nn.ConvTranspose3d(512,256,(1,2,2),stride=(1,2,2)),nn.ConvTranspose3d(256,128,(1,2,2),stride=(1,2,2)),nn.ConvTranspose3d(128,64,(1,2,2),stride=(1,2,2)),nn.ConvTranspose3d(64,32,(1,2,2),stride=(1,2,2))]); self.dec=nn.ModuleList([Block(512,256),Block(256,128),Block(128,64),Block(64,32)]); self.head=nn.Conv3d(32,2,1)
    def forward(self,batch):
        x=F.pad(explicit_sequence(batch),(0,8,0,8)); skips=[]
        for block in self.enc: x=block(x); skips.append(x); x=self.pool(x)
        x=self.center(x)
        for up,block,skip in zip(self.up,self.dec,reversed(skips)):
            x=up(x)
            if x.shape[-3:]!=skip.shape[-3:]: x=F.interpolate(x,size=skip.shape[-3:],mode="trilinear",align_corners=False)
            x=block(torch.cat((skip,x),1))
        prediction=self.head(x)[:,:,:,:200,:200]
        return assert_output(prediction[:,:,1:].permute(0,2,1,3,4).contiguous(),batch)
