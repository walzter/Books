#!/usr/bin/env/python3 

# modified the loss function 

import torch
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    """
    Loss function as seen in: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    L(Y,X1,X2,W) = (1-Y)*(1/2 * D_w **2) + (Y)*(1/2)*max(0, m-D_w)**2
    
    Where D_w = np.sqrt(torch.pow(out1 - out2),2) or the Euclidean distance torch.nn.functional.pairwise_distance(out1, out2)
    
    We initialize with a specified margin, m, which can be either 1 or 2. 
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin 
        
    # the forward pass 
    def forward(self, out1, out2, label):
        """
        """
        # Euclidean distance of the pairs 
        euc_dist = F.pairwise_distance(out1, out2)
        # contrastive loss 
        contrastive_loss = torch.mean((1-label)*torch.pow(euc_dist,2)) + (label) * torch.pow(torch.clamp(self.margin - euc_dist, min=0.0),2)
        closs = contrastive_loss.mean()
        return closs
    
    
# trying with a different loss function 
def contrastive_loss_function(img1, img2, label, m=1.0):
    # do a step by step loss 
    # first the distance 
    distance = img1 - img2
    # squaring it 
    distsqr = torch.sum(torch.pow(distance, 2), 1)
    # squareroot 
    dist = torch.sqrt(distsqr)
    # difference between margin 
    mdist = m - dist 
    # clamp the values 
    dist = torch.clamp(mdist, min=0.0)
    # loss 
    loss = label*distsqr + (1-label) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / img1.size()[0]
    return loss
    