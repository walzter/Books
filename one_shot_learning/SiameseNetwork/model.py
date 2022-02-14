#!/usr/bin/env/python3


# imports 
import torch 
from torch import nn

class SiameseNetwork(nn.Module):
    def __init__(self, weight_init=False): 
        super(SiameseNetwork,self).__init__()

        """
        It will have the following layers 
        
        CONVOLUTION BLOCK:
        ------------------
        Conv2D: (1 channel, 64 features, 10 kernel_size)
        MaxPool2D: (2x2) 
        
        Conv2D: (64 channels, 128 features, 7 kernel_size)
        MaxPool2D: (2x2)
        
        Conv2D: (128 channels, 128 features, 4 kernel_size)
        MaxPool2D: (2x2)

        Conv2D: (128 channels, 256 features, 4 kernel_size)

        FULLY CONNECTED BLOCK:
        ----------------------
        Linear: (256 * 6 * 6, 4096)
        Linear: (4096, 1)
        
        WEIGHT INIT:
        ------------
        we can initialize weight by going through the modules 
        for x in self.modules():
            if isinstance(x, nn.Conv2D):
                nn.init.Glorot_normal(x.weight,mode='fan-in')
                
        """
        # we will have 4 Convolution blocks 
        self.conv1 = nn.Conv2D(1, 64, 10)
        self.conv2 = nn.Conv2D(64,128,7)
        self.conv3 = nn.Conv2D(128,128,4)
        self.conv4 = nn.Conv2D(128,256,4)
        # maxpool always te same
        self.pool = nn.MaxPool2D(2,2)
        # connected layers 
        self.fc1 = nn.Linear(256*6*6,4096)
        self.fc2 = nn.Linear(4096, 1)


        # weight initialization for the Convolutional Layers 
        if weight_init:
            for x in self.modules():
                if isinstance(x, nn.Conv2D):
                    nn.init.kaiming_normal(x, mode='fan-in')

    
    def once_forward(self, x):
        """ single forward throgh one of the network parts 

        FORWARD PASS:
        -------------
        X = F.relu(F.max_pool(self.conv1(x)))
        X = F.relu(F.max_pool(self.conv2(x)))
        X = F.relu(F.max_pool(self.conv3(x)))
        X = F.relu(self.conv4(x))

        # we change the view of the out 
        X = X.view(X.shape[0], -1)

        # passing through the FC Layer

        X = F.sigmoid(self.fc1(out))

        """
        out = F.relu(self.pool(self.conv1(x)))
        out = F.relu(self.pool(self.conv2(out)))
        out = F.relu(self.pool(self.conv3(out)))
        out = F.relu(self.conv4(out))

        # change the view
        out = out.view(out.shape[0], -1)

        # pass through the FC1
        out = F.sigmoid(self.fc1(out))
        
        return out

    def forward(self, x1, x2):
        """ complete forwards through siamese network 
        
        # now with the two parts of the  networks 
        
        out1 = self.forward(x)
        out2 = self.forward(x) 
`       
        # calcualte the distance between them 
        diff = torch.abs(out1 - out2) 

        # present a similarity score here
        score = self.fc2(diff)

        """
        out1 = self.once_forward(x1)
        out2 = self.once_forward(x2)

        # difference 
        diff = torch.abs(out1 - out2)

        # passing through the final FC layer 
        score = self.fc2(diff)
        return score

