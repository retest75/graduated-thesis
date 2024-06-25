import math
import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

from baseline.resnet_modified import CustomizedResNet50
# from model.backbone import CustomizedResnet50
from model.simsiam import SimSiam
from model.classifier import Classifier
from dataset.eval_dataset import EvaluationDataset
from train.ssl_eval import Evaluation, Testing
from focal_loss.focal_loss import FocalLoss

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
batch_size = 128
Epochs = 30
lr = 0.005
momentum = 0.9               # default = 0.9
weight_decay = 0             # default = 0
weights = [[1.0, 5.0]]
training_layer=["fc"]
# phases = ["Fine-tune-1","Fine-tune-2", "Fine-tune-3", "Fine-tune-4", "Fine-tune-5"]


# record setting (設定實驗紀錄的儲存路徑與 log 檔)
record_path = f"/home/chenze/graduated/thesis/record/baseline"

# augmentation
evaluation = [
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
testing = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

# dataset
eval_pth = "/home/chenze/graduated/thesis/dataset/evaluation-Large/denoisy"
test_pth = "/home/chenze/graduated/thesis/dataset/testing-Large/balance-denoisy"
transform = {
    "eval":transforms.Compose(evaluation),
    "test":transforms.Compose(testing)
}
dataset = {
    "eval":EvaluationDataset(eval_pth, transform["eval"], mode="Both"),
    "test":EvaluationDataset(test_pth, transform["test"], mode="Both")
}
dataloader = {
    "eval":DataLoader(dataset["eval"], batch_size=batch_size, shuffle=True),
    "test":DataLoader(dataset["test"], batch_size=batch_size, shuffle=False),
}


since = time.time()
for alpha in weights:
    weight = torch.tensor(alpha).to(device)

    # create folder
    dst = os.path.join(record_path, f"alpha{alpha}") # Ex: /alpha[1.0,20.0]
    os.makedirs(dst, exist_ok=True)

    # load pre-trained model and its weight
    weight_pth = f"/home/chenze/graduated/thesis/record/baseline/pre-train/SL-train-Epoch[04]-Loss[0.164980]-Fscore[0.025](Best).pt"
    param = torch.load(weight_pth)
    model = CustomizedResNet50(weights=ResNet50_Weights.DEFAULT, n_classes=1)
    model.load_state_dict(param)

    # revised pre-trained model
    model = Classifier(model=model, n_classes=1, training_layer=training_layer)
    model = model.to(device)

    # criterion
    criterion = FocalLoss(gamma=1.8, weights=weight)
    # criterion = nn.BCELoss()


    # optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_lambda = lambda epoch: 0.5 * (1 + math.cos(epoch * math.pi / Epochs))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # evaluation
    fine_tune = Evaluation(device, model, dataset["eval"], dataloader["eval"], criterion, optimizer, scheduler)
    testing = Testing(device, model, dataset["test"], dataloader["test"], criterion)

    eval_loss_list = []
    test_loss_list = []
    eval_acc_list = []
    test_acc_list = []
    eval_fscore_list = []
    test_fscore_list = []
    best_loss = float("inf")
    best_fscore = 0
    best_acc = 0
    best_param = None
    best_epoch = None

    print(f"============== Weight: {alpha} ===============")

    epoch_times = 0
    for epoch in range(Epochs):
        eval_loss, eval_acc, eval_fscore, eval_time = fine_tune.eval_fn(epoch)
        test_loss, test_acc, test_fscore, test_time = testing.test_fn()
        
        eval_loss_list.append(eval_loss)
        test_loss_list.append(test_loss)
        eval_acc_list.append(eval_acc)
        test_acc_list.append(test_acc)
        eval_fscore_list.append(eval_fscore)
        test_fscore_list.append(test_fscore)
        epoch_times += (eval_time + test_time)

        # find best parameter
        if test_acc > best_acc:
            best_epoch = epoch
            best_acc = test_acc
            # best_param = fine_tune.model.state_dict()   
            best_param = testing.model.state_dict()     

        # save checkpoint
        if (epoch+1) % 15 == 0:
            fine_tune.save_checkpoint(dst, "Evaluation", epoch)

        # save record
        fine_tune.save_log(os.path.join(dst, "eval-record.log"), "Eval", epoch, Epochs)
        testing.save_log(os.path.join(dst, "test-record.log"), "Test", epoch, Epochs)

        # print evaluation information for each epoch
        print("=" * 20)
        print(f"Epoch: {epoch+1}/{Epochs} for Alpha = {alpha}")
        print(f"Phase: Eval | Loss: {eval_loss:.6f} | F-1 score: {eval_fscore:.4f} | Acc: {eval_acc*100:.3f}% | Times: {eval_time} sec")
        print(f"Phase: Test | Loss: {test_loss:.6f} | F-1 score: {test_fscore:.4f} | Acc: {test_acc*100:.3f}% | Times: {test_time} sec")
        print("=" * 20)

        # save testing loss and fscore
        testing.save_loss(os.path.join(dst, "test-loss.log"))
        testing.save_fscore(os.path.join(dst, "test-fscore.log"))
        testing.save_acc(os.path.join(dst, "test-acc.log"))
    
    print(f"Alpha: {alpha} and Complete ! Times: {epoch_times//3600} hr {epoch_times//60%60} min {epoch_times%60} sec")

    # save best parameter and entire training loss, F-1 score
    fine_tune.save_checkpoint(dst, "eval", best_epoch, best_param)
    fine_tune.save_loss(os.path.join(dst, "eval-loss.log"))
    fine_tune.save_fscore(os.path.join(dst, "eval-fscore.log"))
    fine_tune.save_acc(os.path.join(dst, "eval-acc.log"))
    

    # plot loss, learning rate, and f-score
    plt.plot(range(1, Epochs+1), eval_loss_list, label="eval")
    plt.plot(range(1, Epochs+1), test_loss_list, label="test")
    plt.legend()
    plt.title(f"Loss with Alpha = {alpha}")
    plt.savefig(os.path.join(dst, "loss.png"))
    plt.clf()
    
    plt.plot(range(1, Epochs+1), fine_tune.lr, label="Learning rate")
    plt.legend()
    plt.title(f"Learning Rate with Alpha = {alpha}")
    plt.savefig(os.path.join(dst, "learning-rate.png"))
    plt.clf()
    
    plt.plot(range(1, Epochs+1), eval_acc_list, label="eval")
    plt.plot(range(1, Epochs+1), test_acc_list, label="test")
    plt.legend()
    plt.title(f"Acc with Alpha = {alpha}")
    plt.savefig(os.path.join(dst, "acc.png"))
    plt.clf()

    plt.plot(range(1, Epochs+1), eval_fscore_list, label="eval")
    plt.plot(range(1, Epochs+1), test_fscore_list, label="test")
    plt.legend()
    plt.title(f"F-1 score with Alpha = {alpha}")
    plt.savefig(os.path.join(dst, "fscore.png"))
    plt.clf()

training_times = int(time.time() - since)
print(f"End, Times: {training_times//3600} hr {training_times//60%60} min {training_times%60} sec")

