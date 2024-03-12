import time
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from dataset.create import Create_dataset
from baseline.resnet_modified import CustomizedResNet50




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
    with torch.no_grad():
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


if __name__ == "__main__":
    root = "../dataset"
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_ds = Create_dataset(root=root, transform=transform, mode="Left", stage="train")
    test_ds = Create_dataset(root=root, transform=transform, mode="Left", stage="test")
    #count = 0
    #for img, lbl in DataLoader(dataset, batch_size=100, shuffle=False):
    #    print(f"img size: {img.size()} | lbl = {lbl}, size = {lbl.size()}")
    #    count += img.size(0)
    #print(count)
    batsh_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    model = CustomizedResNet50(weights=weights, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 2

    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    since = time.time()
    for epoch in range(epochs):
        print(f"Epochs: {epoch+1}/{epochs}")
        print("-" * 15)
        train_loss, train_acc, _ = train_fn(dataset=train_ds, batch_size=batsh_size, device=device, model=model, optimizer=optimizer, criterion=criterion)
        test_loss,test_acc = test_fn(dataset=test_ds, batch_size=batsh_size, device=device, model=model, optimizer=optimizer, criterion=criterion)
        print(f"Phase: training | Loss: {train_loss:.6f} | Acc: {100 * train_acc:.2f}%")
        print(f"Phase: validate | Loss: {test_loss:.6f} | Acc: {100 * test_acc:.2f}%")
        print()

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
    
    times = int(time.time() - since)
    print(f"training time: {times//3600}h {times//60%60}min {times%60}sec")



