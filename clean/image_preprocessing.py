# A module of preprocessing for image
# It contains read, crop and then save the croped image

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Image_preprocess():
    """
    Parameter
    ------------
    src : source directory name and path
    dst : destinated directory name and path
    """
    def __init__(self, src: str, dst: str):

        self.src = src
        self.dst = dst

        # 思考要在物件內部直接進行還是實體化後再進行
        #for index, filename in enumerate(os.listdir(self.src)):
        #    img = self.read_img(os.path.join(self.src, filename))
        #    img = self.crop_img(img)
        #    self.save_img(img, os.path.join(self.dst, str(index)+".jpg"))
    
    def read_img(self, img_path: str):
        img = cv2.imread(img_path)                 # [H, W, C]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB

        return img
    
    def crop_img(self, img: np.ndarray):
        img_croped = img[360:1500, 420:1884, :] # [2304, 1536, 3] -> [1464, 1464, 3]

        return img_croped
    
    def save_img(self, img: np.array, dst: str):
        plt.imsave(dst, img)
    



if __name__ == "__main__":
    src = "C:\\graduated\\thesis\\data\\dataset\\testing-image"
    dst = None

    # 思考要在物件內部直接進行還是實體化後再進行
    #ip = Image_preprocess()
    #for index, filename in enumerate(os.listdir(src)):
    #    img = ip.read_img(os.path.join(src, filename))
    #    img = ip.crop_img(img)
    #    ip.save_img(img, os.path.join(dst, str(index)+".jpg"))
    




