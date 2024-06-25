import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader

from dataset.eval_dataset import EvaluationDataset
from model.simsiam import SimSiam
# from model.backbone import CustomizedResnet50
from baseline.resnet_modified import CustomizedResNet50
from model.classifier import Classifier
from focal_loss.focal_loss import FocalLoss
from train.ssl_eval import Testing

# basic configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
param = "eval-Epoch[02]-Loss[0.196968]-Fscore[0.096](Best)" + ".pt"
weight = torch.tensor([1.0, 5.0], device=device)

# record setting (設定實驗紀錄的儲存路徑與 log 檔)
path = f"/home/chenze/graduated/thesis/record/baseline/Linear(1.8)"
record_path = f"/home/chenze/graduated/thesis/record/baseline/Linear(1.8)/test"
os.makedirs(record_path, exist_ok=True)

# augmentation
augmentation = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

# dataset
root = "/home/chenze/graduated/thesis/dataset/testing-Large/balance-denoisy"
transform = transforms.Compose(augmentation)
dataset = EvaluationDataset(root, transform, mode="Both")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

 # load pre-trained model and revised it
# simsiam = SimSiam(CustomizedResnet50())
model = CustomizedResNet50(weights=ResNet50_Weights.DEFAULT, n_classes=1)
model = Classifier(model=model, n_classes=1)
# model = Classifier(model=simsiam.encoder[0].resnet, n_classes=1)

# load weight
weight_pth = os.path.join(path, param)
param = torch.load(weight_pth)
model.load_state_dict(param)
model = model.to(device)

# criterion
# criterion = nn.BCELoss()
criterion = FocalLoss(gamma=2.0, weights=weight)

# testing
testing = Testing(device, model, dataset, dataloader, criterion)

if __name__ == "__main__":
    test_loss, test_acc, test_fscore, _ = testing.test_fn()

    # plot confusion matrix
    testing.confusion_matrix(record_path)

    # plot P-R curve
    testing.compute_pr_curve(record_path)

    # plot ROC curve
    testing.compute_roc(record_path)

    # save indicator
    testing.save_indicator(os.path.join(record_path, "indicator.log"))

    print(f"Loss: {test_loss:.6f}")
    print(f"Acc: {test_acc*100:.2f}%")
    print(f"F-1 score: {test_fscore:.2f}")
    print(f"Precision: {testing.compute_precision():.2f}")
    print(f"Recall: {testing.compute_recall():.2f}")
    print(f"AU-ROC: {testing.auroc:.2f}")
    print(f"AU-PRC: {testing.auprc:.2f}")

    

