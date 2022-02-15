import random 
import numpy as np 
import matplotlib.pyplot as plt

def make_pairs(images, positive_indices):
    # holders 
    positive_imgs = []
    negative_imgs = []
    pair_labels   = []
    # get the length of the smallest group 
    min_len = min([len(positive_indices[k]) for k in range(10)]) - 1 # -1 to keep it within bounds 
    # now we loop through the number of classes 
    for i in range(10):
        # now in range of the smallest lenght 
        for k in range(min_len):
            pos1, neg1 = positive_indices[i][k], positive_indices[i][k+1] # we are in the same class & choose a different sample
            positive_imgs.append(images[pos1].numpy())
            negative_imgs.append(images[neg1].numpy())
            pair_labels.append([1])
            # now for the pairs 
            # we create a random increment 
            rand_increment = random.randrange(1,10)
            # this random increment gets added to our class 
            rand_idx = (i + rand_increment) % 10 # to see the remainder and not equal to itself 
            # we can get the other pairs 
            pos2, neg2 = positive_indices[i][k], positive_indices[rand_idx][k]
            positive_imgs.append(images[pos2].numpy())
            negative_imgs.append(images[neg2].numpy())
            pair_labels.append([0])
    # now convert them all to arrays & reshape
    # reshape into corresponding one batch 
    positive_imgs = np.array(positive_imgs, dtype=np.float32)
    positive_imgs = positive_imgs / 255.
    positive_imgs = positive_imgs.reshape([-1,1,28,28])
    # negative
    negative_imgs = np.array(negative_imgs, dtype=np.float32)
    negative_imgs = negative_imgs / 255.
    negative_imgs = negative_imgs.reshape([-1,1,28,28])
    # labels 
    pair_labels = np.array(pair_labels, dtype=np.int32)
    
    return positive_imgs, negative_imgs, pair_labels
    
# plotting the loss
def plot_loss (loss, label=None):
    plt.plot(loss, label=label)
    plt.legend()
    plt.show()
