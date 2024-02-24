# each of fundus files has duplicate, for example, xxxxxxxx_L.jpg and xxxxxxxx_L(1).jpg
# so we 
# (1) copy these repetivite files to another folder (duplicated_manth) then compare which files are best
# (2) after choosing the best one, we revise these filename and move back to original folder
# (3) delete these repetitive file i.e., xxxxxxxx_L(1).jpg
# (4) split entire dataset into left and right eyes

#################################################################
#                                                               #
#               正式使用時,記得把初始化函數中的方法解開            #
#                                                               #
#################################################################

import os
import shutil


class Clean():
    """ 
    Attribute
    ------------
    self.path       : folder path for each of years
    self.duplicated : folder path including duplicated files
    self.months     : list which save all months
    self.len        : size of dataset
    """

    def __init__(self, root, year):
        self.path = os.path.join(root, year)
        self.duplicated = os.path.join(self.path, "duplicated")
        self.months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

        #self.duplicate()          # copy repetitive file into another folder
        #for month in self.months:
        #    self.move_back(month) # rename repetitive file and move back to original folder
        #self.delete()             # rename repetitive file
        self.left_right()         # separate into left and right eye
        self.len = self.length(folder="Left") + self.length(folder="Right")
        
    def length(self, folder):
        """ calculate size of dataset """

        path = os.path.join(self.path, folder) # /year/folder
        return len(os.listdir(path))

    def duplicate(self):
        """ copy repetitive files to another folder """
        
        for month in self.months:
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

    def left_right(self):
        """ split images into left and right eyes """

        left_folder = os.path.join(self.path, "Left")   # year/Left
        right_folder = os.path.join(self.path, "Right") # year/Right

        for month in self.months:
            src_folder = os.path.join(self.path, month) # year/month          
            filename_list = os.listdir(src_folder)

            for filename in filename_list:
                if filename[-5] == "L":
                    src = os.path.join(src_folder, filename)   # year/month/xxxxxxxxx_L.jpg
                    dst = os.path.join(left_folder, filename)  # year/Left/xxxxxxxxx_L.jpg
                else:
                    src = os.path.join(src_folder, filename)   # year/month/xxxxxxxxx_R.jpg
                    dst = os.path.join(right_folder, filename) # year/Right/xxxxxxxxx_L.jpg
                shutil.move(src, dst)


if __name__ == "__main__":
    root = "C:\\graduated\\thesis\\data"
    years = ["2022"]

    for year in years:
        clean = Clean(root, year)
        print(f"Year: {year} Complete, size = {clean.len}")     
    print()
    print("===== Clean complete =====")
