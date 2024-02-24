# 1. 整理「眼底鏡and睡眠呼吸中止症」以及「呼吸中止症報表」
#    檢驗兩者是否有高度重疊與包含關係
# 2. 整理「眼底鏡and睡眠呼吸中止症」以及「眼底鏡匯出報表」
#    檢驗兩者是否有高度重疊與包含關係
# 3. 列出各年度眼底影像資料集的完整長度

import os
import pandas as pd
import numpy as np

class Check():
    """ 
    1. 整理「眼底鏡and睡眠呼吸中止症」以及「呼吸中止症報表」, 檢驗兩者是否有高度重疊與包含關係
    2. 整理「眼底鏡and睡眠呼吸中止症」以及「眼底鏡匯出報表」, 檢驗兩者是否有高度重疊與包含關係
    3. 列出各年度所有報表與資料集的數量
    """
    def __init__(self, root, year):
        """
        Variable
        ----------
        root: root directory included each years
        year: the year we want to clean and ckeck

        Attribute
        ----------
        self.month: months for each years
        self.path: path for each year
        self.report_path: path for each report, i.e., fundus(眼底鏡匯出報表), sleep(睡眠中止症匯出報表), fun_sl(眼底and中止症匯出報表)
        
        """
        self.months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        self.path = os.path.join(root, year)
        self.report_path_fundus = os.path.join(self.path, "fundus_" + year + ".xls")
        self.report_path_sleep = os.path.join(self.path, "sleep_" + year + ".xls")
        self.report_path_fun_slp = os.path.join(self.path, "fundus_and_sleep_" + year + ".xls")

        self.compare()

    def fundus(self):
        fun_report = pd.read_excel(self.report_path_fundus)

        self.fundus_No = []
            
        for number in fun_report["去個資編號"]:
            if pd.isna(number):
                continue
            else:
                self.fundus_No.append(number)

        #print(f"Length of fundus: {len(self.fundus_No)}")
            
        return self.fundus_No
        
    def sleep(self):
        slp_report = pd.read_excel(self.report_path_sleep)

        self.sleep_No = []
            
        for number in slp_report["去個資編號"]:
           if pd.isna(number):
                continue
           else:
               self.sleep_No.append(number)

        #print(f"Length of fundus: {len(self.sleep_No)}")
            
        return self.sleep_No
        
    def fun_slp(self):
        fun_slp_report = pd.read_excel(self.report_path_fun_slp)

        self.fun_slp_No = []
            
        for number in fun_slp_report["去個資編號"]:
            if pd.isna(number):
                continue
            else:
                self.fun_slp_No.append(number)

        #print(f"Length of fundus: {len(self.fundus_slp_No)}")
            
        return self.fun_slp_No
    
    def total_img(self):
        index = 0

        for month in self.months:
            path = os.path.join(self.path, month)
            files_list = os.listdir(path)
            index = index + len(files_list)
        
        return index
        
    def compare(self):
        a = set(self.fundus())
        b = set(self.sleep())
        c = set(self.fun_slp())

        #print("----- Size -----")
        #print(f"Size of fundus report: {len(a)}")
        #print(f"Size of sleep report: {len(b)}")
        #print(f"Size of fundus and sleep report: {len(c)}")
        #print(f"Total fundus images: {self.total_img()}")
        #print()

        #print("----- Association -----")
        #print(f"(fundus) & (fundus and sleep ): {len(a&c):2d}")
        #print(f"(sleep) & (fundus and sleep ):  {len(b&c):2d}")
        return a, b, c


        
if __name__=="__main__":
    root = "C:\\graduated\\thesis\\data"
    years = ["2022", "2021", "2020", "2019", "2018"]

    print(f"| Year | Fundus report(A) | Sleep report(B) | Fundus and Sleep report(C) | A & C | B & C | Total images |")
    
    for year in years:
        #print(f"Year: {year}")
        check = Check(root, year)
        a, b, c = check.compare()
        print(f"| {year} |      {len(a):4d}        |       {len(b):3d}       |           {len(c):2d}               |  {len(a&c):2d}   |  {len(b&c):2d}   |    {check.total_img():5d}     |")
        
        #print()