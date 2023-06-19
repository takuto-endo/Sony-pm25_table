
""" 公式 discription
提供データを読み込み、前処理を施し、モデルに入力が可能な状態に変換するモジュール。
get_train_dataやget_test_dataのように、学習用と評価用を分けて、前処理を行う関数を定義。
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


class Preprocessing(object):
    """docstring for Preprocessing"""
    def __init__(self):
        super(Preprocessing, self).__init__()
        self.cat_encoders = {}

    def get_train_data(self):
        print("\n[CHECK POINT]: START \"get_train_data\"\n")

        categories = []
        categorical_dims = {}

        train_df = pd.read_csv("../data/train.csv")

        # カテゴリカルデータ変換
        cat_encoders = {}
        for col in train_df.columns:
            if col=="City":
                break
            if train_df[col].dtype=="object":
                cat_encoder = LabelEncoder()
                train_df[col] = cat_encoder.fit_transform(train_df[col])
                categories.append(col)
                categorical_dims[col] = len(cat_encoder.classes_)
                self.cat_encoders[col] = cat_encoder

        print("shape of train data: ", train_df.shape)
        if len(categories)==0:
            categories = None
        print("categories: ", categories)
        print("\n[CHECK POINT]: END \"get_train_data\"\n")
        return train_df, categories, categorical_dims

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

if __name__ == "__main__":
    pass 