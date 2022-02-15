#!/usr/bin/env/python3

# defining the siamese dataset
class SiameseDataset(object):
    def __init__(self, image1, image2, label):
        self.image1 = image1
        self.image2 = image2
        self.label  = label
        self.size   = label.shape[0]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        im1 = self.image1[index]
        im2 = self.image2[index]
        y   = self.label[index]
        return (im1, im2, y)
