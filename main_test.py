import torch
from torch.utils.data import DataLoader

def train_fn(dataset, batch_size, device, model, optimizer, criterion):
    """ an simple training phase framswork for an epochs """

    batch_loss = 0
    batch_correct = 0

    model.train()
    for img, lbl in DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True):
        img, lbl = img.to(device), lbl.to(device)

        output = model(img)
        
        optimizer.zero_grad()
        loss = criterion(output, lbl)

        _, pred = torch.max(output, 1)

        loss.backward()
        optimizer.step()

        batch_loss += loss.item() * img.size(0)
        batch_correct += torch.sum(pred == lbl.data)
    
    epoch_loss = batch_loss / len(dataset)
    epoch_acc = batch_correct.float() / len(dataset)
    epoch_param = model.state_dict()

    return epoch_loss, epoch_acc, epoch_param

def test_fn(dataset, batch_size, device, model, optimizer, criterion):
    """ an eacy testing phase framefork for each epochs """
    
    batch_loss = 0
    batch_correct = 0

    model.eval()
    with torch.no_grad:
        for img, lbl in DataLoader(dataset, batch_size, shuffle=False):
            img, lbl = img.to(device), lbl.to(device)

            output = model(img)
            loss = criterion(output, lbl)

            _, pred = torch.max(output, 1)
            
            batch_loss += loss.item() * img.size(0)
            batch_correct += torch.sum(pred == lbl.data)
        
    epoch_loss = batch_loss / len(dataset)
    epoch_acc = batch_correct.float() / len(dataset)

    return epoch_loss, epoch_acc
        