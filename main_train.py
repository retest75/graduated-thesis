import math
import os
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from model.backbone import CustomizedResnet50
from model.simsiam import SimSiam
from dataset.pre_trained_dataset import GaussianBlur, TwoCropTransforms, PreTrainedDataset
from train.ssl_train import Training

#----- Environment information -----#
print("========== Environment information ==========")
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Device name: {torch.cuda.get_device_name()}")
print()
#-----------------------------------#
print("============== Start Training ===============")


    
# basic configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
Epochs = 100
lr_base = 0.5                      # default = 0.05
lr = (lr_base * batch_size) / 256  # default = 0.025
# lr = 0.00625 # decay 5 times
momentum = 0.9                     # default = 0.9
weight_decay = 0.0001              # default = 0.0001

# record setting (設定實驗紀錄的儲存路徑與 log 檔)
record_path = "/home/chenze/graduated/thesis/record/SimSiam(ResNet50)"
phase = "Pre_train"
model_name = "SimSiam(ResNet50)"
optimizer_name = "SGD"


# augmentation
augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur()], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

# dataset
root = "/home/chenze/graduated/thesis/dataset/pre-trained"
transform = transforms.Compose(augmentation)
dataset = PreTrainedDataset(root, TwoCropTransforms(transform), mode="both", classes=2)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model
model = SimSiam(CustomizedResnet50())
model = model.to(device)

# criterion
criterion = nn.CosineSimilarity()

# optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_lambda = lambda epoch: 0.5 * (1 + math.cos(epoch * math.pi / Epochs))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# training
pre_trained = Training(device, model, dataset, dataloader, criterion, optimizer, scheduler)

history_loss = []
history_time = 0
best_loss = float("inf")
best_param = None
best_epoch = None
    
if __name__ == "__main__":
    for epoch in range(Epochs):
        epoch_loss, epoch_time = pre_trained.train_fn(epoch)

        history_loss.append(epoch_loss)
        history_time += epoch_time

        # find best parameter
        if epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_loss
            best_param = pre_trained.model.state_dict()

        # save checkpoint
        if (epoch+1) % 40 == 0:
            pre_trained.save_checkpoint(record_path, phase, epoch)

        # save record
        pre_trained.save_log(os.path.join(record_path, "record.log"), model_name, phase, optimizer_name, epoch, Epochs)

        # print training information for each epoch
        print("=" * 20)
        print(f"Epoch: {epoch+1}/{Epochs} | Loss: {epoch_loss:.6f} | Times: {epoch_time} sec")
        print("=" * 20)
    
    # save best parameter and entire training loss
    pre_trained.save_checkpoint(record_path, "Pre_train", best_epoch, best_param)
    pre_trained.save_loss(os.path.join(record_path, "loss.log"))
    print(f"Training Complete ! Times: {history_time//3600} hr {history_time//60%60} min {history_time%60} sec")

    # plot loss and learning rate
    plt.plot(range(1, Epochs+1), history_loss, label="Loss")
    #plt.xticks(range(1, Epochs+1))
    plt.legend()
    plt.title(f"Pre-trained Loss of {model_name} using {optimizer_name}")
    plt.savefig(os.path.join(record_path, "loss.png"))
    
    plt.clf()
    plt.plot(range(1, Epochs+1), pre_trained.lr, label="Learning rate")
    #plt.xticks(range(1, Epochs+1))
    plt.legend()
    plt.title(f"Pre-trained Learning Rate of {model_name} using {optimizer_name}")
    plt.savefig(os.path.join(record_path, "learning-rate.png"))
