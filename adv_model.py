import torch.nn as nn
from torch import cat
import torch

class AdvBacterialClassiferCNN(nn.Module):
    def __init__(self, 
                 HW,
                 conv1,
                 k1,
                 s1,
                 K_pool1,
                 conv2,
                 k2,
                 s2,
                 K_pool2,
                 conv3,
                 k3,
                 s3,
                 K_pool3,
                 fc1,
                 fce,
                 meta_feats,
                 out_dim,
                in_channels=3
                ):
        super().__init__()
        #Convolution block 1
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=conv1,kernel_size=k1,padding='same',stride=s1)
        self.mpool1 = nn.MaxPool2d(kernel_size=K_pool1)
        self.act1 = nn.SiLU()
        HW_mpool1 = self.calc_convolution_dims(HW=HW,kernel=K_pool1,padding=0,stride=K_pool1)

        #Convolution block 2
        self.conv2 = nn.Conv2d(in_channels=conv1,out_channels=conv2,kernel_size=k2,stride=s2,padding='same')
        self.act2 = nn.SiLU()
        self.bn2d1 = nn.BatchNorm2d(conv2)
        self.mpool2 = nn.MaxPool2d(K_pool2)
        HW_mpool2 = self.calc_convolution_dims(HW=HW_mpool1,kernel=K_pool2,stride=K_pool2)

        #Convolution block 3
        self.conv3 = nn.Conv2d(in_channels=conv2,out_channels=conv3,kernel_size=k3,stride=s3,padding='same')
        self.act3 = nn.SiLU()
        self.drop2d1 = nn.BatchNorm2d(conv3)
        self.mpool3 = nn.MaxPool2d(K_pool3)
        HW_mpool3 = self.calc_convolution_dims(HW=HW_mpool2,kernel=K_pool3,stride=K_pool3)
    
        #output layer
        self.flatten = nn.Flatten()
        flattened_dim = HW_mpool3*HW_mpool3*conv3 # H * W * C
        self.fc1 = nn.Linear(flattened_dim,fc1)
        self.drop1d1 = nn.Dropout1d()

        #metadata
        self.fce = nn.Linear(meta_feats,fce)
        self.acte = nn.SiLU()

        self.fco = nn.Linear((fc1+fce),out_dim) #output layer

    def calc_convolution_dims(self,HW:int,kernel:int,padding:int=0,stride:int=1):
        new_HW_dim = ((HW + 2*padding - kernel)/stride)+1
        return int(new_HW_dim)

    def forward(self, im, meta:torch.Tensor):
        #convolution block 1
        im = self.conv1(im)
        im = self.act1(im)
        im = self.mpool1(im)
        
        #convolution block 2
        im = self.conv2(im)
        im = self.act2(im)
        im = self.bn2d1(im)
        im = self.mpool2(im)
        
        #convolution block 2
        im = self.conv3(im)
        im = self.act3(im)
        im = self.drop2d1(im)
        im = self.mpool3(im)

        #output
        im = self.flatten(im)
        im = self.fc1(im)
        im = self.drop1d1(im)

        #embedding
        meta = meta.to(dtype=torch.float32)
        meta = self.fce(meta)
        meta = self.acte(meta)

        #out
        comb = cat((im,meta),dim=1)
        logits = self.fco(comb)
        return logits