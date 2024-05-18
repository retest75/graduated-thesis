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
from model.classifier import Classifier
from dataset.eval_dataset import EvaluationDataset
from train.ssl_train import Evaluation

#---------- README ----------#
# this eval used to fine-tune or linear probe
# (1) before training, setup "basic configuration            (Line: 27)
# (2) before training, setup "record setting"                (Line: 36)
# (3) before training, setup dataset path in dataset comment (Line: 50)
# (4) before training, setup "save checkpoint" comment       (Line: 105)
# (5) if want to change other backbone, revise model comment (Line: 71)
#----------------------------#

# basic configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
Epochs = 10
lr_base = 0.03 #default = 0.03
lr = (lr_base * batch_size) / 256
momentum = 0.9
weight_decay = 0

# record setting (設定實驗紀錄的儲存路徑與 log 檔)
record_path = "/home/chenze/graduated/thesis/record/SimSiam(ResNet50)"
phase = "Linear"
model_name = "SimSiam(ResNet50)"
optimizer_name = "SGD"

# augmentation
augmentation = [
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

# dataset
root = "/home/chenze/graduated/thesis/dataset/evaluation-Large/balance"
transform = transforms.Compose(augmentation)
dataset = EvaluationDataset(root, transform, mode="Both")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

 # load pre-trained model and its weight
weight_pth = "/home/chenze/graduated/thesis/record/SimSiam(ResNet50)/Ex-1/2/Pre_train-Epoch[72]-Loss[-0.971076](Best).pt"
param = torch.load(weight_pth)
simsiam = SimSiam(CustomizedResnet50())
simsiam.load_state_dict(param)

# revised pre-trained model
model = Classifier(model=simsiam.encoder[0].resnet, n_classes=2, phase="Linear")
model = model.to(device)

# criterion
criterion = nn.CrossEntropyLoss()

# optimizer and scheduler
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_lambda = lambda epoch: 0.5 * (1 + math.cos(epoch * math.pi / Epochs))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# evaluation
linear_probe = Evaluation(device, model, dataset, dataloader, criterion, optimizer, scheduler)

history_loss = []
history_acc = []
history_fcore = []
history_time = 0
# best_loss = float("inf")
best_fscore = -float("inf")
best_param = None
best_epoch = None

#----- Environment information -----#
print("========== Environment information ==========")
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Device name: {torch.cuda.get_device_name()}")
print()
#-----------------------------------#
print("============== Start Training ===============")

if __name__ == "__main__":

    for epoch in range(Epochs):
        epoch_loss, epoch_acc, epoch_fscore, epoch_time = linear_probe.eval_fn(epoch)

        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)
        history_fcore.append(epoch_fscore)
        history_time += epoch_time

        # find best parameter
        # if epoch_loss < best_loss:
        #     best_epoch = epoch
        #     best_loss = epoch_loss
        #     best_param = linear_probe.model.state_dict()
        if epoch_fscore > best_fscore:
            best_epoch = epoch
            best_fscore = epoch_fscore
            best_param = linear_probe.model.state_dict()        

        # save checkpoint
        if (epoch+1) % 10 == 0:
            linear_probe.save_checkpoint(record_path, phase, epoch)

        # save record
        linear_probe.save_log(os.path.join(record_path, "record.log"), model_name, phase, optimizer_name, epoch, Epochs)

        # print training information for each epoch
        print("=" * 20)
        print(f"Epoch: {epoch+1}/{Epochs} | Loss: {epoch_loss:.6f} | F-1 score: {epoch_fscore:.2f} | Acc: {epoch_acc*100:.2f}% | Times: {epoch_time} sec")
        print("=" * 20)
    
    # save best parameter and entire loss, F-1 score, and acc
    linear_probe.save_checkpoint(record_path, phase, best_epoch, best_param)
    linear_probe.save_loss(os.path.join(record_path, "loss.log"))
    linear_probe.save_acc(os.path.join(record_path, "acc.log"))
    print(f"Training Complete ! Times: {history_time//3600} hr {history_time//60%60} min {history_time%60} sec")

    # plot loss, learning rate and acc
    plt.plot(range(1, Epochs+1), history_loss, label="Loss")
    plt.legend()
    plt.title(f"Linear-probe Loss of {model_name} using {optimizer_name}")
    plt.savefig(os.path.join(record_path, "loss.png"))
    
    plt.clf()
    plt.plot(range(1, Epochs+1), linear_probe.lr, label="Learning rate")
    plt.legend()
    plt.title(f"Linear-probe Learning Rate of {model_name}")
    plt.savefig(os.path.join(record_path, "learning-rate.png"))
    
    plt.clf()
    plt.plot(range(1, Epochs+1), history_acc, label="Acc")
    plt.plot(range(1, Epochs+1), history_fcore, label="F-1 score")
    plt.title(f"Linear-probe Acc of {model_name} using {optimizer_name}")
    plt.savefig(os.path.join(record_path, "acc.png"))