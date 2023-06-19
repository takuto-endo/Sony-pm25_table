
""" 公式 discription
提供データを読み込み、前処理を施し、モデルに入力が可能な状態に変換するモジュール。
get_train_dataやget_test_dataのように、学習用と評価用を分けて、前処理を行う関数を定義。
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Preprocessing(object):
    """docstring for Preprocessing"""
    def __init__(self):
        super(Preprocessing, self).__init__()
        self.cat_encoders = {}

    def get_train_data(self):
        print("\n[CHECK POINT]: START \"get_train_data\"\n")

        categories = []
        # categorical_dims = {}
        categorical_dims = []
        continuous = []

        train_df = pd.read_csv("../data/train.csv")

        # カテゴリカルデータ変換
        cat_idxs = []
        con_idxs = []
        cat_encoders = {}
        unused_fea = ["City", "id"]
        target_col = "pm25_mid"
        for i, col in enumerate(train_df.drop(unused_fea+[target_col], axis=1).columns):
            if train_df[col].dtype=="object":
                cat_encoder = LabelEncoder()
                train_df[col] = cat_encoder.fit_transform(train_df[col])
                categories.append(col)
                # categorical_dims[col] = len(cat_encoder.classes_)
                categorical_dims.append(len(cat_encoder.classes_))
                self.cat_encoders[col] = cat_encoder

                cat_idxs.append(i)
            else:
                continuous.append(col)
                con_idxs.append(i)

        print("shape of train data: ", train_df.shape)
        if len(categories)==0:
            categories = None
            cat_idxs = None
        print("categories: ", categories)

        print("\n[CHECK POINT]: END \"get_train_data\"\n")

        return train_df, con_idxs, continuous, cat_idxs, categories, categorical_dims# categories: カテゴリ列名, categorical_dims: カテゴリの次元

    def get_test_data(self):
        
        print("\n[CHECK POINT]: START \"get_test_data\"\n")
        
        test_df = pd.read_csv("../data/test.csv")
        test_df = test_df.drop("City", axis=1)
        
        # カテゴリカルデータ変換
        for col in test_df.columns:
            if test_df[col].dtype=="object":
                test_df[col] = self.cat_encoders[col].transform(test_df[col])

        print("shape of test data: ", test_df.shape)
        print("\n[CHECK POINT]: END \"get_test_data\"\n")
        return test_df

class DataSetCatCon(Dataset):
    def __init__(self, X, mask, Y, cat_idxs, con_idxs, task='reg', continuous_mean_std=None):

        cat_cols = list(cat_idxs)
        con_cols = list(con_idxs)

        self.X1 = np.array(X.iloc[:,cat_cols].copy().astype(np.int64)) #categorical columns
        self.X2 = np.array(X.iloc[:,con_cols].copy().astype(np.float32)) #numerical columns
        self.X1_mask = np.array(mask[:,cat_cols].copy().astype(np.int64)) #categorical columns: maskの雛形生成
        self.X2_mask = np.array(mask[:,con_cols].copy().astype(np.int64)) #numerical columns: maskの雛形生成
        # print("X1: ", self.X1.shape)
        # print("X1_mask: ", self.X1_mask.shape)
        # print("X2: ", self.X2.shape)
        # print("X2_mask: ", self.X2_mask.shape)

        if task == 'clf':
            self.y = np.array(Y)#.astype(np.float32)
        else:
            self.y = np.array(Y.astype(np.float32))
        # print("y: ", self.y.shape)
        self.cls = np.zeros_like(self.y,dtype=int)# 出力と同じsize
        # print("cls: ", self.cls.shape)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        # print("cls_mask: ", self.cls_mask.shape)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
        # print("X2: ", self.X2.shape)

    def __len__(self):# Dataset必須アイテム
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]


class TestDataSetCatCon(Dataset):
    def __init__(self, X, mask, cat_idxs, con_idxs, task='reg', continuous_mean_std=None):

        cat_cols = list(cat_idxs)
        con_cols = list(con_idxs)

        # print("X: ", X.shape)

        self.X1 = np.array(X.iloc[:,cat_cols].copy().astype(np.int64)) #categorical columns
        self.X2 = np.array(X.iloc[:,con_cols].copy().astype(np.float32)) #numerical columns
        self.X1_mask = np.array(mask[:,cat_cols].copy().astype(np.int64)) #categorical columns: maskの雛形生成
        self.X2_mask = np.array(mask[:,con_cols].copy().astype(np.int64)) #numerical columns: maskの雛形生成

        # print("X1: ", self.X1.shape)
        # print("X1_mask: ", self.X1_mask.shape)
        # print("X2: ", self.X2.shape)
        # print("X2_mask: ", self.X2_mask.shape)
        self.cls = np.zeros_like(self.X1[:,0],dtype=int)# 出力と同じsize
        self.cls = self.cls.reshape(-1,1)
        # print("cls: ", self.cls.shape)
        self.cls_mask = np.ones_like(self.X1[:,0],dtype=int)
        self.cls_mask = self.cls_mask.reshape(-1,1)
        # print("cls_mask: ", self.cls_mask.shape)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):# Dataset必須アイテム
        return len(self.cls)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]



if __name__ == "__main__":
    pass 