import torch.nn as nn
import torch

class FaceNet(nn.Module):
    def __init__(self,in_channels,hidden_size,output_shape):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_size,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size*50*50,out_features=output_shape)  
        )
    def forward(self,x):
        x = self.conv_block1(x)
        x =self.conv_block2(x)
        x = torch.sigmoid(self.classifier(x))
        return x