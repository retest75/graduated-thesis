import os
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.eval_dataset import EvaluationDataset
from model.simsiam import SimSiam
from model.backbone import CustomizedResnet50
from model.classifier import Classifier

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# basic configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
pre_trained = "Pre-trained-1"
phase = "Linear-5"
alpha = [1.0, 5.0]
gamma = 1.8
# record setting (設定實驗紀錄的儲存路徑與 log 檔)
path = f"/home/chenze/graduated/thesis/record/design-2/{pre_trained}/{phase}/alpha{alpha}/gamma[{gamma}]"
record_path = f"/home/chenze/graduated/thesis/record/design-2/{pre_trained}/{phase}/alpha{alpha}/gamma[{gamma}]/test"
os.makedirs(record_path, exist_ok=True)

# augmentation
augmentation = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

# dataset
root = "/home/chenze/graduated/thesis/dataset/testing-Large/balance-denoisy/1_disease"
for side in ["Left", "Right"]:
    pth = os.path.join(root, side)

    for filename in os.listdir(pth):
        file_pth = os.path.join(pth, filename)

        transform = transforms.Compose(augmentation)
        # dataset = EvaluationDataset(root, transform, mode="Both")
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        img = Image.open(file_pth).convert("RGB")
        img = img.resize((223, 224))
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_array = np.array(img)
        img_array = np.float32(img_array)/255

        # load pre-trained model and revised it
        simsiam = SimSiam(CustomizedResnet50())
        model = Classifier(model=simsiam.encoder[0].resnet, n_classes=1)
        output = model(img_tensor)
        pred = F.sigmoid(output)


        # load weight
        weight_pth = os.path.join(path, "eval-Epoch[08]-Loss[0.172099]-Fscore[0.453](Best).pt")
        param = torch.load(weight_pth)
        model.load_state_dict(param)
        # model = model.to(device)
        target_layers = [model.model.layer4[-1]]

        # Construct the CAM object once, and then re-use it on many images
        cam = GradCAM(model=model, target_layers=target_layers)

        grayscale_cam = cam(input_tensor=img_tensor, targets=None)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True, image_weight=0.7)

        if pred.item() >= 0.5:
            os.makedirs(os.path.join(record_path, "1"), exist_ok=True)
            cv2.imwrite(os.path.join(record_path, "1", filename), visualization)
        else:
            os.makedirs(os.path.join(record_path, "0"), exist_ok=True)
            cv2.imwrite(os.path.join(record_path, "0", filename), visualization)
    

