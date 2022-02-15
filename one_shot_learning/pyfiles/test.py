#!/usr/bin/env/python3
import torch 
import numpy as np

def test_model(model, loader):
    # put into eval mode
    model.eval()
    # holders
    output, all_labels = [],[]
    # no grads in case
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(loader):
            label = label.float()
            # get the outputs 
            out = model.once_forward(image)
            # extending the list with the output of the model
            output.extend(out.data.cpu().numpy().tolist())
            # extending the labels 
            all_labels.extend(label.data.cpu().numpy().tolis())
    # converting to numpy 
    n_out = np.array(output)
    n_labels = np.array(all_labels)
    return n_out, n_labels
            
            