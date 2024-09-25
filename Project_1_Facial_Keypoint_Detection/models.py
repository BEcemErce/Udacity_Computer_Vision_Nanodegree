## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5,padding=1)
        self.pool1=nn.MaxPool2d(2, 2) 
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1=nn.Dropout(0.1)
        
        
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool2=nn.MaxPool2d(2, 2) 
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2=nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 4, padding=1)
        self.pool3=nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3=nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4=nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4=nn.Dropout(0.4)

        
        self.fc1=nn.Linear(256 * 13 * 13,640)
        #self.bn5 = nn.BatchNorm1d(1000)
        self.drop5=nn.Dropout(0.5)
        
        self.fc2=nn.Linear(640,768)
        #self.bn6 = nn.BatchNorm1d(1000)
        self.drop6=nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(768, 136)
        
        #I used initiliazer for the fc as namishNet paper suggested.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool1(F.relu(self.conv1(x)))# I used relu function for the activation
        x=self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x=self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x=self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x=self.drop4(x)
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x=self.drop5(x)
        x = F.relu(self.fc2(x))
        x=self.drop6(x)
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
    
    
    

