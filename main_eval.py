import math
import os
import time
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
# from train.ssl_train import Evaluation
from train.ssl_eval import Evaluation, Testing
from loss.focal_loss import BinaryFocalLoss
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
Epochs = 50
# lr_base = 0.003                    # default = 0.03
# lr = (lr_base * batch_size) / 256 # default = 0.015
lr = 0.005
momentum = 0.9                    # default = 0.9
weight_decay = 0.0001             # default = 0
gamma_search = [2.0, 1.8, 1.6, 1.4, 1.2]
weights = [
    [1.0,20.0],
    [1.0,16.0],
    [1.0,12.0],
    [1.0,10.0]
]

# record setting (設定實驗紀錄的儲存路徑與 log 檔)
record_path = "/home/chenze/graduated/thesis/record/design-2"

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
eval_pth = "/home/chenze/graduated/thesis/dataset/evaluation-Large"
test_pth = "/home/chenze/graduated/thesis/dataset/testing-Large/imbalance-2(1.5-1)"
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

    for gamma in gamma_search:
        # create folder
        dst = os.path.join(record_path, f"alpha{alpha}", f"gamma[{gamma}]") # Ex: /alpha[1.0,20.0]/gamma[2.0]
        os.makedirs(dst, exist_ok=True)

        # load pre-trained model and its weight
        weight_pth = "/home/chenze/graduated/thesis/record/design-2/Ex-1-PreTrained/2/Pre_train-Epoch[100]-Loss[-0.940438](Best).pt"
        param = torch.load(weight_pth)
        simsiam = SimSiam(CustomizedResnet50())
        simsiam.load_state_dict(param)

        # revised pre-trained model
        model = Classifier(model=simsiam.encoder[0].resnet, n_classes=1, phase=None)
        model = model.to(device)

        # criterion
        criterion = FocalLoss(gamma=gamma, weights=weight)
        # criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)

        # optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
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
        best_fscore = -float("inf")
        best_param = None
        best_epoch = None

        print(f"============== Weight: {alpha} and Gamma: {gamma} ===============")

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
            if test_loss < best_loss:
                best_epoch = epoch
                best_loss = test_loss
                best_param = fine_tune.model.state_dict()        

            # save checkpoint
            if (epoch+1) % 10 == 0:
                fine_tune.save_checkpoint(dst, "Evaluation", epoch)

            # save record
            fine_tune.save_log(os.path.join(dst, "eval-record.log"), "Eval", epoch, Epochs)
            testing.save_log(os.path.join(dst, "test-record.log"), "Test", epoch, Epochs)

            # print evaluation information for each epoch
            print("=" * 20)
            print(f"Epoch: {epoch+1}/{Epochs} for Gamma = {gamma}, Alpha = {alpha}")
            print(f"Phase: Eval | Loss: {eval_loss:.6f} | F-1 score: {eval_fscore:.4f} | Acc: {eval_acc*100:.3f}% | Times: {eval_time} sec")
            print(f"Phase: Test | Loss: {test_loss:.6f} | F-1 score: {test_fscore:.4f} | Acc: {test_acc*100:.3f}% | Times: {test_time} sec")
            print("=" * 20)

            # save testing loss and fscore
            testing.save_loss(os.path.join(dst, "test-loss.log"))
            testing.save_fscore(os.path.join(dst, "test-fscore.log"))
            testing.save_acc(os.path.join(dst, "test-acc.log"))
        
        print(f"Alpha: {alpha} and Gamma: {gamma} Complete ! Times: {epoch_times//3600} hr {epoch_times//60%60} min {epoch_times%60} sec")

        # save best parameter and entire training loss, F-1 score
        fine_tune.save_checkpoint(dst, "eval", best_epoch, best_param)
        fine_tune.save_loss(os.path.join(dst, "eval-loss.log"))
        fine_tune.save_fscore(os.path.join(dst, "eval-fscore.log"))
        fine_tune.save_acc(os.path.join(dst, "eval-acc.log"))
        

        # plot loss, learning rate, and f-score
        plt.plot(range(1, Epochs+1), eval_loss_list, label="eval")
        plt.plot(range(1, Epochs+1), test_loss_list, label="test")
        plt.legend()
        plt.title(f"Loss with Gamma = {gamma}, Alpha = {alpha}")
        plt.savefig(os.path.join(dst, "loss.png"))
        plt.clf()
        
        plt.plot(range(1, Epochs+1), fine_tune.lr, label="Learning rate")
        plt.legend()
        plt.title(f"Learning Rate with Gamma = {gamma}, Alpha = {alpha}")
        plt.savefig(os.path.join(dst, "learning-rate.png"))
        plt.clf()
        
        plt.plot(range(1, Epochs+1), eval_acc_list, label="eval")
        plt.plot(range(1, Epochs+1), test_acc_list, label="test")
        plt.legend()
        plt.title(f"Acc with Gamma = {gamma}, Alpha = {alpha}")
        plt.savefig(os.path.join(dst, "acc.png"))
        plt.clf()

        plt.plot(range(1, Epochs+1), eval_fscore_list, label="eval")
        plt.plot(range(1, Epochs+1), test_fscore_list, label="test")
        plt.legend()
        plt.title(f"F-1 score with Gamma = {gamma}, Alpha = {alpha}")
        plt.savefig(os.path.join(dst, "fscore.png"))
        plt.clf()

training_times = int(time.time() - since)
print(f"End, Times: {training_times//3600} hr {training_times//60%60} min {training_times%60} sec")

