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
        # data["model"] = data["model"].fillna(0).astype("int64",errors="ignore")
    return data

fname_train = os.path.join(data_dir, "used_car_train_20200313.zip")
data_train = extract_data(fname_train)
fname_testA = os.path.join(data_dir, "used_car_testA_20200313.zip")
data_testA = extract_data(fname_testA)


def correct_type(df):
    """
    （1）from float64 to int64：'model', 'bodyType', 'fuelType', 'gearbox'
        convert NA to -1
    （2）from object to int64：'notRepairedDamage'
        convert - to -1
    （3）from int64 to float64：'power'
    """
    for col in ['model', 'bodyType', 'fuelType', 'gearbox']:
        df[col] = df[col].fillna(-1).astype("int64")
    df.loc[df["notRepairedDamage"]=="-","notRepairedDamage"] = -1
    df["notRepairedDamage"] = df["notRepairedDamage"].astype("float64").astype("int64")
    df["power"] = df["power"].astype("float64")
    return df

data_train = correct_type(data_train)
data_testA = correct_type(data_testA)


def process_date(df):
    """
    regDate， creatDate
    """
    df["regYear"] = df["regDate"].astype("str").str.slice(0,4).astype("int64")
    df["regMonth"] = df["regDate"].astype("str").str.slice(4,6).astype("int64")
    df["regDay"] = df["regDate"].astype("str").str.slice(6,8).astype("int64")
    df["creatYear"] = df["creatDate"].astype("str").str.slice(0, 4).astype("int64")
    df["creatMonth"] = df["creatDate"].astype("str").str.slice(4, 6).astype("int64")
    df["creatDay"] = df["creatDate"].astype("str").str.slice(6, 8).astype("int64")
    return df

data_train = process_date(data_train)
data_testA = process_date(data_testA)


cols_drop = ["name", "regDate", "creatDate"]
trainy = data_train.pop("price").astype("float64")
trainX = data_train.drop(cols_drop, axis=1)
testAX = data_testA.drop(cols_drop, axis=1)

