import os
import numpy as np
import cv2

class Image_preprocess():
    def __init__(self, path):
        self.path = path

        print(f"file path: {self.path}")

        for filename in os.listdir(self.path):
            pth = os.path.join(self.path, filename) # file path
            img = self.read_img(pth)
            img = self.crop(img)

    def read_img(self, img_path):
        """ Read image and transform to RGB mode """

        img = cv2.imread(img_path)                 # [H, W, C]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
        print(f"Original shape = [{img.shape[0]}, {img.shape[1]}, {img.shape[2]}]")
        
        return img

    def crop(self, img):
        """ Crop image to square """

        img = img[:, 384:1920, :]
        print(f"Cropped shape = [{img.shape[0]}, {img.shape[1]}, {img.shape[2]}]")
        return img



if __name__ == "__main__":
    root = "/home/chenze/graduated/thesis/dataset"
    condition = ["1_disease"]
    direction = ["Left", "Right"]

    for c in condition:
        for d in direction:
            path = os.path.join(root, c, d)
            preprocess = Image_preprocess(path)





