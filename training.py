import time
import torch
import torchvision
import torch.optim
# import the transformer model here

def get_data_loader(dataset: torchvision.datasets, batch_size=32, splits=[0.8,0.1,0.1]) -> torch.utils.data.DataLoader:
    '''
    dataset: torchvision.datasets a transformer dataset for training, testing, and validation
    batch_size: int
    splits: list(str) train-validation-test split
    return: DataLoader
    '''

    assert sum(splits) == 1, "ensure sum of train-validation-test split adds up to 1"

    # perform split
    size = len(dataset)
    l1, l2 = int(size*splits[0]), int(size*splits[1])
    l3 = size - l1 - l2

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [l1, l2, l3],
        generator=torch.Generator().manual_seed(999)
    )

    # get data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, val_loader, test_loader

def train(data_loader, model, loss_function, optimizer, scheduler=None, epochs=30): 
    losses_over_epochs = []
    num_batches = len(data_loader)

    for epoch in epochs:
        start = time.time()
        total_loss = 0
        for (source, target, labels, source_mask, target_mask) in data_loader:
            # forward step
            out = model(source, target, source_mask, target_mask)

            # loss
            loss = loss_function(out, labels)
            total_loss += loss.item()

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # learning rate scheduler update
        if scheduler is not None:
            scheduler.step()

        # finished one epoch of training
        end = time.time()
        print(f"Completed epoch {epoch+1} | average loss: {total_loss/num_batches} | time: {end-start}s")
        losses_over_epochs.append(total_loss/num_batches)

    return losses_over_epochs