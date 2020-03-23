# coding:utf-8

import zipfile
import os
import numpy as np
import pandas as pd

data_dir = r"data/"

def extract_data(fname):
    with zipfile.ZipFile(fname) as zfile:
        data_file = zfile.open(zfile.namelist()[0])
        data = pd.read_csv(data_file, sep=" ", index_col=0)
        data["model"] = data["model"].astype("int64",errors="ignore")
    return data

fname_testA = os.path.join(data_dir, "used_car_testA_20200313.zip")
data_testA = extract_data(fname_testA)
fname_train = os.path.join(data_dir, "used_car_train_20200313.zip")
data_train = extract_data(fname_train)


