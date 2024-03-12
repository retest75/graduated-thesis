# Create.py Note
# 1. Purpose: create a customized dataset to DataLoader
# 2. architecture
#  (1) in __init__() method, read image path and save into self.img
#  (2) in __init_() method, read label and save into self.lbl
# 3. Usage
#  we use cv2 to read and crop in __getitem__() method and it will return numpy array
#  so we must transfor numpy into PIL format (use transforms.ToPILImage()) since transforms must input a PIL image


##### if you want test this script, unlock this block #####
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#---------------------------------------------------------#

import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from clean.image_preprocessing import Image_preprocess




class Create_dataset(Dataset):
    def __init__(self, root, transform=None, mode=None, stage="train"):
        """ Create a customized dataset 
        
        Parameter
        ------------
        root      : root directory
        transform : transformation sequence
        mode      : str, "Both", "Left" or "Right"

        Attribute
        ------------
        self.img       : list, save each image path
        self.lbl       : list, save label information corresponding to each image
        self.root      : root directory
        self.transform : transformation sequence
        self.sides     : list, save selection mode for eyes direction
        self.classes   : dict, save folder name and its label

        Method
        ------------
        __getitem__ : get image according to index
        __len__     : length of dataset
        __sides     : decide both or single eyes
        """
        self.img = [] # image path in sequence
        self.lbl = [] # label in sequence

        self.root = os.path.join(root, stage)
        self.transform = transform

        self.sides = self.__sides(mode)
        self.classes = {"0_normal":0, "1_disease":1}

        for classes in self.classes.keys():
            for side in self.sides:
                path = os.path.join(self.root, classes, side)
                
                for filename in os.listdir(path):
                    self.img.append(f"{path}/{filename}")
                    self.lbl.append(self.classes[classes])
    
    def __getitem__(self, index):
        img_pth = self.img[index]
        lbl = self.lbl[index]
        
        preprocess = Image_preprocess(path=None)
        img = preprocess.read_img(img_pth)
        img = preprocess.crop(img)

        ##### if Image_preprocess module(above) can not use, uncommentt this block to use cv2 #####
        #img = cv2.imread(img_pth)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img[36:1500, 420:1884, :]
        #-----------------------------------------------------------------------------------------#

        if self.transform != None:
            img = self.transform(img)
        
        return img, lbl
    
    def __len__(self):
        return len(self.img)
    
    def __sides(self, mode):
        """ decide both or single eyes 
        
        Variable
        ------------
        mode : str, "Both", "Left", or "Right"

        Return
        ------------
        [] : list, single side for ["Left"] or ["Right"] and both side for ["Left", "Right"]
        """
        if (mode == "Left") or (mode == "Right"):
            return [mode]
        else:
            return ["Left", "Right"]




if __name__ == "__main__":
    root = "/home/chenze/graduated/thesis/dataset"
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = Create_dataset(root=root, transform=transform, mode="Left", stage="test")
    
    for (img, lbl) in DataLoader(dataset=dataset, batch_size=100):
        print(f" {img.size()} | {lbl.size()}")




