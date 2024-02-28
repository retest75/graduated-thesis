# each of fundus files has duplicate, for example, xxxxxxxx_L.jpg and xxxxxxxx_L(1).jpg
# so we 
# (1) copy these repetivite files to another folder (duplicated_manth) then compare which files are best
# (2) after choosing the best one, we revise these filename and move back to original folder
# (3) delete these repetitive file i.e., xxxxxxxx_L(1).jpg
# (4) split entire dataset into left and right eyes
#  after spliting, we label all image
# (1) list each of years filename for particular report where 1: disease and 0: normal
# (2) list image filename in entire dataset for certain year
# (3) compare and take intersection these two lists
# (4) move intersection file to destination folder

import os
import shutil
import numpy as np
import pandas as pd


class Clean():
    """ 
    Variable
    ------------
    root  : str, path for root folder
    years : list, all years wanted to process

    Attribute
    ------------
    self.years      : list, all years wanted to process
    self.months     : list, which save all months
    self.path       : root/year
    self.duplicated : root/year/duplicated
    self.len        : size of dataset
    """

    def __init__(self, root, years):
        self.root = root
        self.years = years
        self.months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        
        print("===== Data clean =====")
        for year in self.years:
            self.path = os.path.join(self.root, year)                    # year
            self.duplicated = os.path.join(self.path, "duplicated") # year/duplicated

            for month in self.months:
                self.duplicate(month) # copy repetitive file into another folder
                self.move_back(month) # rename repetitive file and move back to original folder
                self.delete()         # rename repetitive file
                self.left_right(self.path)     # separate into left and right eye
                self.len = self.length(folder="Left") + self.length(folder="Right")
            print(f"Year: {year} complete, size = {self.len:5d}")
        print("===== Clean complete, start classify =====")
        print()        
        
    def length(self, folder):
        """ calculate size of dataset """

        path = os.path.join(self.path, folder) # /year/folder
        return len(os.listdir(path))

    def duplicate(self, month):
        """ copy repetitive files to another folder """
        
        src_folder = os.path.join(self.path, month)                       # year/month
        dst_folder = os.path.join(self.duplicated, "duplicated_" + month) # year/duplicated/duplicated_xx
        filename_list = os.listdir(src_folder)

        for filename in filename_list:
            if filename[-5] == ')':
                src = os.path.join(src_folder, filename) # year/month/xxxxxxxxx_L(1).jpg
                dst = os.path.join(dst_folder, filename) # year/duplicated/duplicated_xx/xxxxxxxxx_L(1).jpg
                shutil.copyfile(src, dst)
    
    def filename_process(self, filename):
        """ process repetitive filename in particular month """

        filename, _ = filename.split('(')
        filename = filename + ".jpg"
        
        return filename
    
    def move_back(self, month):
        """ move revised filename back to original folder """

        dst_folder = os.path.join(self.path, month)                       # year/month
        src_folder = os.path.join(self.duplicated, "duplicated_" + month) # year/duplicated/duplicated_xx
        src_filename = os.listdir(src_folder)

        for old_name in src_filename:
            new_name = self.filename_process(old_name) # xxxxxxxxx_L(1).jpg -> xxxxxxxxx_L.jpg
            src = os.path.join(src_folder, old_name)   # year/duplicated/duplicated_xx/xxxxxxxxx_L(1).jpg
            dst = os.path.join(dst_folder, new_name)   # year/month/xxxxxxxxx_L.jpg
            #shutil.copyfile(src, dst)
            shutil.move(src, dst)
    
    def delete(self):
        """  delete repetitive files """

        for month in self.months:
            dst_folder = os.path.join(self.path, month) # year/month
            filename_list = os.listdir(dst_folder)

            for filename in filename_list:
                if filename[-5] == ')':
                    path = os.path.join(dst_folder, filename) # year/month/xxxxxxxxx_L(1).jpg
                    os.remove(path)

    def left_right(self, path):
        """ split images into left and right eyes """

        left_folder = os.path.join(path, "Left")   # year/Left
        right_folder = os.path.join(path, "Right") # year/Right

        for month in self.months:
            src_folder = os.path.join(path, month) # year/month          
            filename_list = os.listdir(src_folder)

            for filename in filename_list:
                if filename[-5] == "L":
                    src = os.path.join(src_folder, filename)   # year/month/xxxxxxxxx_L.jpg
                    dst = os.path.join(left_folder, filename)  # year/Left/xxxxxxxxx_L.jpg
                    shutil.move(src, dst)
                elif filename[-5] == "R":
                    src = os.path.join(src_folder, filename)   # year/month/xxxxxxxxx_R.jpg
                    dst = os.path.join(right_folder, filename) # year/Right/xxxxxxxxx_R.jpg
                    shutil.move(src, dst)
                else:
                    continue
                   

class Inquiry(Clean):
    """ Inherit Clean to make the entire dataset become smaller and label it

    Attribute
    ------------
    self.side    : left or right eyes
    self.reports : report name and  its label

    """

    def __init__(self, root, years):
        super().__init__(root, years)

        self.sides = {"Left":"L", "Right":"R"}
        self.reports = {"fundus_and_sleep":1, "fundus":0}

        for report_name in self.reports.keys():
            for side in self.sides.keys():
                report = self.report_list(report_name, self.sides[side]) # return all filename in some report

                for year in self.years:
                    img = self.img_list(year, side) # return all img filename in some year and side
                    intersec = self.take_intersec(report, img)
                    self.move(year, self.reports[report_name], side, intersec)
            print(f"Report: {report_name} complete, label = {self.reports[report_name]}")
        print("===== Classify complete =====")

    def report_list(self, report_name, side):
        """ list all filename in some report """

        filename_list = []
        
        for year in self.years:
            report = report_name + "_" + year + ".xls"   # report filename
            path = os.path.join(self.root, year, report) # /year/fundus_xxxx.xls
            df = pd.read_excel(path)

            for idx in list(df["眼底鏡影像去個資編號"]):
                if np.isnan(idx):
                    continue
                else:
                    name = str(int(idx)) + "_" + side + ".jpg" # xxxxxxxxx_L.jpg
                    filename_list.append(name)
        return filename_list
    
    def img_list(self, year, side):
        """ list all filename in some year """
        
        path = os.path.join(self.root, year, side) # /year/Left
        return os.listdir(path)
    
    def take_intersec(self, report_list, img_list):
        """ take intersection between report and all dataset """

        return list(set(report_list) & set(img_list))
    
    def move(self, year, mode, side, filename_list):
        """ move file from original to dataset """

        classify = {0:"0_normal", 1:"1_disease"}

        for file in filename_list:
            src = os.path.join(self.root, year, side, file) # /year/Left/xxxxxxxxx_X.jpg
            dst = os.path.join(self.root, "dataset", classify[mode], side, file) # /dataset/0_normal/Left/xxxxxxxxx_X.jpg
            shutil.move(src, dst)


if __name__ == "__main__":
    root = "C:\\graduated\\thesis\\data"
    years = ["2022", "2021", "2020", "2019", "2018"]

    #clean = Clean(root, years)
    inquiry = Inquiry(root, years)

    
