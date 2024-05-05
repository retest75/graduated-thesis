# This dataset be used to pre-trained model of SimSiam Network
# It include an Gaussian Blur augmentation which did not be contain in PyTorch
# This dataset will produce two views for one image
# 使用方式參照下方 if __name__ == "__main__" 的設計
# 務必要經過 TwoCropTransforms() 才會產生兩個視圖
# Augmentation reference
#  (1) RandomGrayscale : https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomGrayscale.html#torchvision.transforms.RandomGrayscale
#  (2) ColorJitter     : https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter
#  (3) RandomApply     : https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomApply.html#torchvision.transforms.RandomApply

import os
import random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

class PreTrainedDataset(Dataset):
    def __init__(self, root, transform, mode="both"):
        super().__init__()

        self.transform = transform
        self.img = [] # save all image path
        self.lbl = [] # save all label name

        for side in self._sides(mode):
            path = os.path.join(root, side)

            for filename in os.listdir(path):
                self.img.append(f"{path}/{filename}")
                self.lbl.append(side)
    
    def __getitem__(self, index):
        """
        The return type is a list which include two item.
        Each item is a [B, C, H, W] tensor, you can use img[0] and img[1] to see each views of an image.
        """
        img_path = self.img[index]
        lbl_name = self.lbl[index]

        img = Image.open(img_path)
        img = img.convert("RGB")

        # TwoCropTransform
        img_list = self.transform(img)

        return img_list, lbl_name
     
    def __len__(self):
        return len(self.lbl)

    def _sides(self, mode):
        """ Decide which eyes will be used to pre-trained model """
        if mode == "Left":
            return ["Left"]
        
        elif mode == "Right":
            return ["Right"]
        
        else:
            return ["Left", "Right"]


class TwoCropTransforms():
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)

        return [x1, x2]


class GaussianBlur():
    def __init__(self, sigma=[.1, .2]):
        self.sigma = sigma
    
    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        return img


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms

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

    transform = transforms.Compose(augmentation)
    root = "C:\\graduated\\thesis\\data\\dataset\\pre_trained"
    dataset = PreTrainedDataset(root, TwoCropTransforms(transform), mode="both")
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
#    for (idx, (img, lbl)) in enumerate(dataloader):
#        if idx <= 5:
#            print(img[0].shape, img[1].shape)
#            print()
    print(f"dataset length: {dataset.__len__()}")