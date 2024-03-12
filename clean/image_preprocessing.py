# a module of preprocessing for image
# input a folder name which include images
# then you can use read_img method to read it and crop method to crop it to square-sized
# you can utilize some for loops in __init__ method to read and crop also

import os
import cv2

class Image_preprocess():
    def __init__(self, path):
        self.path = path

    def read_img(self, img_path):
        """ Read image and transform to RGB mode """

        img = cv2.imread(img_path)                 # [H, W, C]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
        #print(f"Original shape = [{img.shape[0]}, {img.shape[1]}, {img.shape[2]}]")
        
        return img

    def crop(self, img):
        """ Crop image to square where orig shape = [1536, 2304, 3] """

        #img = img[:, 384:1920, :]       # shape: [H W, C] = [1536, 1536, 3]
        img  = img[36:1500, 420:1884, :] # shape: [H W, C] = [1464, 1464, 3]
        #print(f"Cropped shape = [{img.shape[0]}, {img.shape[1]}, {img.shape[2]}]")
        return img



if __name__ == "__main__":
    root = "/Users/ChenZE/graduated/thesis/dataset/0_normal/Left/137988984_L.jpg"
    #condition = ["1_disease"]
    #direction = ["Left", "Right"]

    #for c in condition:
    #    for d in direction:
    #        path = os.path.join(root, c, d)
    #        preprocess = Image_preprocess(path)
    preprocess = Image_preprocess(path=None)
    img = preprocess.crop(preprocess.read_img(root))
    print(img.shape)




