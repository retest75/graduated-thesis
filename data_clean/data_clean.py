# each of fundus files has duplicate, for example, xxxxxxxx_L.jpg and xxxxxxxx_L(1).jpg
# so we copy these repetivite files to another folder (duplicated_manth) then compare which files are best
# after choosing the best one, we revise these filename and move back to original folder
# finally, we delete these repetitive file i.e., xxxxxxxx_L(1).jpg

import os
import shutil


class Clean():
    """ 
    Attribute
    ------------
    self.path : folder path for each of years
    self.duplicated : folder path including duplicated files
    self.months : list which save all months
    """
    def __init__(self, root, year):
        self.path = os.path.join(root, year)
        self.duplicated = os.path.join(self.path, "duplicated")
        self.months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

        self.duplicate()
        for month in self.months:
            self.move_back(month)
        self.delete()
        
    def duplicate(self):
        """ copy repetitive files to another folder """
        
        for month in self.months:
            src_folder = os.path.join(self.path, month)                       # source
            dst_folder = os.path.join(self.duplicated, "duplicated_" + month) # destination

            filename_list = os.listdir(src_folder)

            for (idx, filename) in enumerate(filename_list):
                if filename[-5] == ')':
                    src = os.path.join(src_folder, filename)
                    dst = os.path.join(dst_folder, filename)
                    shutil.copyfile(src, dst)
    
    def filename_process(self, filename):
        """ process repetitive filename for particular month """

        filename, _ = filename.split('(')
        filename = filename + ".jpg"
        
        return filename
    
    def move_back(self, month):
        """ move revised filename back to original folder """

        dst_folder = os.path.join(self.path, month)
        src_folder = os.path.join(self.duplicated, "duplicated_" + month)
        src_filename = os.listdir(src_folder)

        for old_name in src_filename:
            new_name = self.filename_process(old_name)
            src = os.path.join(src_folder, old_name)
            dst = os.path.join(dst_folder, new_name)
            shutil.copyfile(src, dst)
    
    def delete(self):
        """  delete repetitive files """

        for month in self.months:
            dst_folder = os.path.join(self.path, month)
            filename_list = os.listdir(dst_folder)

            for filename in filename_list:
                if filename[-5] == ')':
                    path = os.path.join(dst_folder, filename)
                    os.remove(path)


root = "C:\\graduated\\thesis\\data"
years = ["2022"]

if __name__ == "__main__":
    for year in years:
        clean = Clean(root, year)
        print(f"Year: {year} Complete !")
    



