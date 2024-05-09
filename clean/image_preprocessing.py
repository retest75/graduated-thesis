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
    root  : str, root directory, default: /dataset
    """
    def __init__(self, root: str):

        self.root = root # /dataset/
        self.count_1 = 0 # count for [2304, 1536, 3]
        self.count_2 = 0 # count for [1536, 2298, 3]

    def get_image_path(self, mode:str =None, label:int =None):
        """
        Parameter
        ------------
        mode  : setup "pre_trained" for pre_train dataset or "None" for labeled dataset
        label : 0 for normal or 1 for disease, only setup when mode is None
        """
        if mode == "pre_trained":
            sides = ["Left", "Right"]
            path = os.path.join(self.root, mode)    # /dataset/pre_trained
            path_list = []

            for side in sides:
                sub_path = os.path.join(path, side) # /dataset/pre_trained/Left

                for filename in os.listdir(sub_path):
                    path_list.append(os.path.join(sub_path, filename))

            return path_list
        
        else:
            label_dict = {0:"0_normal", 1:"1_disease"}
            sides = ["Left", "Right"]
            path = os.path.join(self.root, label_dict[label]) # Ex: /dataset/0_normal
            path_list = []

            for side in sides:
                sub_path = os.path.join(path, side)           # Ex: /dataset/0_normal/Left

                for filename in os.listdir(sub_path):
                    path_list.append(os.path.join(sub_path, filename))
            
            return path_list

    def read_img(self, img_path: str):
        img = cv2.imread(img_path)                 # [H, W, C]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB

        return img
    
    def check_size(self, img: np.ndarray):
        if img.shape == (1536, 2304, 3):
            self.count_1 += 1 # count how many imgs whose shape is [1536, 2304, 3]

        elif img.shape == (1536, 2298, 3):
            self.count_2 += 1 # count how many imgs whose shape is [1536, 2298, 3]

        else:
            print(f"other size: {img.shape}")

    def crop_img(self, img: np.ndarray):
        if img.shape == (1536, 2304, 3):
            img_croped = img[36:1500, 420:1884, :] # [1536, 2304, 3] -> [1464, 1464, 3]
            return img_croped
        
        elif img.shape == (1536, 2298, 3):
            img_croped = img[36:1500, 417:1881, :] # [1536, 2298, 3] -> [1464, 1464, 3]
            return img_croped
        
        else:
            pass
    
    def save_img(self, img: np.ndarray, dst: str):
        plt.imsave(dst, img)
    



if __name__ == "__main__":
    import time
    since = time.time()

    root = "C:\\graduated\\thesis\\data\\dataset"
    ip = Image_preprocess(root)

    path_list = ip.get_image_path(mode="pre_trained")

    for img_path in path_list:
        img = ip.read_img(img_path)
        img_crop = ip.crop_img(img)
        ip.save_img(img_crop, img_path)
    
    times = int(time.time() - since)
    print(f"Times: {times//3600} hr {times//60%60} min {times%60} sec")

    
    




