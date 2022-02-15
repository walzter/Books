# now we can start out loop 
def train_model(epochs, model, loader,criteria, optim, batch_size, debug_batch=True):
    """ Trains the SiameseNetwork
    """    
    train_loss = []
    for epoch in range(epochs):
        # set the training of the model
        model.train()
        # iterate through the bathes 
        for idx_batch, (img1, img2, labels) in enumerate(loader):
            labels = labels.float()
            # passing through the forward 
            out1, out2 = model(img1, img2)
            # getting the loss
            loss = criteria(out1, out2, labels)
            # zero gradients 
            optim.zero_grad()
            # backwards pass 
            loss.backward()
            # step 
            optim.step()
            # checking the loss 
            train_loss.append(loss.item())
            if debug_batch:
                if idx_batch % batch_size == 0: 
                    print(f"EPOCH: {epoch} | Loss: {loss.item()}")
    return train_loss