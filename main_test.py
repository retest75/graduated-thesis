import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.eval_dataset import EvaluationDataset
from model.simsiam import SimSiam
from model.backbone import CustomizedResnet50
from model.classifier import Classifier
# from train.ssl_train import Testing
from loss.focal_loss import FocalLoss, BinaryFocalLoss
from train.ssl_eval import Testing

# basic configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128

# record setting (設定實驗紀錄的儲存路徑與 log 檔)
record_path = "/home/chenze/graduated/thesis/record/design-2"
phase = "Testing"
model_name = "SimSiam(ResNet50)"
optimizer_name = "SGD"

# augmentation
augmentation = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

# dataset
root = "/home/chenze/graduated/thesis/dataset/testing-Large/balance"
transform = transforms.Compose(augmentation)
dataset = EvaluationDataset(root, transform, mode="Both")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

 # load pre-trained model and revised it
simsiam = SimSiam(CustomizedResnet50())
model = Classifier(model=simsiam.encoder[0].resnet, n_classes=1)

# load weight
weight_pth = "/home/chenze/graduated/thesis/record/design-2/0/eval-Epoch[01]-Loss[0.016332]-Fscore[0.089](Best).pt"
param = torch.load(weight_pth)
model.load_state_dict(param)
model = model.to(device)

# criterion
# criterion = nn.CrossEntropyLoss()
criterion = BinaryFocalLoss(alpha=[0.05, 1], gamma=2)

# testing
testing = Testing(device, model, dataset, dataloader, criterion)

if __name__ == "__main__":
    loss, acc, f_score, _ = testing.test_fn()

    # plot confusion matrix
    testing.confusion_matrix(pth=record_path)

    # compute f1-score
    # f_score = testing.compute_fscore()

    # plot P-R curve
    testing.compute_pr_curve(pth=record_path)

    # plot ROC curve
    testing.compute_roc(pth=record_path)

    print(f"Loss: {loss:.6f}")
    print(f"Acc: {acc*100:.2f}%")
    print(f"F-1 score: {f_score:.2f}")
    

