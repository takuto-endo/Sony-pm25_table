
""" 公式 discription
Preprocessingの出力を読み込み、モデルを学習し、学習済みモデルを出力するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数等を定義。
"""

from TabNet_Preprocessing import *
from TabNet import *

import time
import glob
from pathlib import Path
import argparse
import os
import shutil

import numpy as np
np.random.seed(seed=0)
import pandas as pd
from sklearn.model_selection import GroupKFold

def learning(num_folds=5, same_reference_path=None):

    parser = argparse.ArgumentParser(description='sony_pm25')
    parser.add_argument('--results_path', type=Path, default=Path('../results'), help='result dir name')
    args = parser.parse_args()
    
    if same_reference_path!=None:
        args.save_path = args.results_path / same_reference_path
    else:
        args.save_path = args.results_path / time.strftime("%Y%m%d-%H%M%S")
        create_exp_dir(args.save_path, scripts_to_save=glob.glob('*.py'))

    preprocessing = Preprocessing()

    train_df, categories, categorical_dims = preprocessing.get_train_data()
    test_df = preprocessing.get_test_data()

    train_X = train_df.drop("pm25_mid", axis=1)
    train_y = train_df["pm25_mid"]
    groups = train_df["City"]

    # モデルの定義
    tabnet = TabNet(args)
    tabnet.pretrain(train_X)

    # Cross Validation
    group_kf = GroupKFold(n_splits=num_folds)
    for random_seed, (train_index, val_index) in enumerate(group_kf.split(train_X, train_y, groups)):
        print("\n[CHECK POINT]: START ", random_seed+1, "\"fold\"\n")
        print("train city: ", train_X.iloc[train_index]["City"].unique())
        X_train = train_X.iloc[train_index].drop(["id", "City"], axis=1).reset_index(drop=True)
        y_train = pd.DataFrame(np.array(train_y.iloc[train_index]).reshape(-1,1), columns=['pm25_mid']).reset_index(drop=True)
        print("val city: ", train_X.iloc[val_index]["City"].unique())
        X_valid = train_X.iloc[val_index].drop(["id", "City"], axis=1).reset_index(drop=True)
        y_valid = pd.DataFrame(np.array(train_y.iloc[val_index]).reshape(-1,1), columns=['pm25_mid']).reset_index(drop=True)
        
        tabnet.train(X_train, y_train, X_valid, y_valid, categories=categories, categorical_dims=categorical_dims, print_importance=True, random_seed=random_seed)
    
        print("\n[CHECK POINT]: END \"", random_seed+1, "fold\"\n")

    print(f"\n[CHECK POINT]: END \"whole learning of tabnet.\" mean_of_train_time={np.mean(tabnet.train_times)}\n")

    return args.save_path, test_df

def create_exp_dir(path, scripts_to_save=None):
    path.mkdir(parents=True, exist_ok=True)
    print("\n[CHECK POINT]: Run dir >> ",path,"\n")

    if scripts_to_save is not None:
        scripts_dir = path / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)

        for script in scripts_to_save:
            dst_file = scripts_dir / os.path.basename(script)
            shutil.copyfile(script, dst_file)

if __name__ == "__main__":
    _, _ = learning()