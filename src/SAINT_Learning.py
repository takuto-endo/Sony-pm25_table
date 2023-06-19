
""" 公式 discription
Preprocessingの出力を読み込み、モデルを学習し、学習済みモデルを出力するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数等を定義。
"""

from SAINT_Preprocessing import *
from SAINT_wrap import *

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

from torch.utils.data import DataLoader

def learning(num_folds=5, same_reference_path=None):

    parser = argparse.ArgumentParser(description='sony_pm25')
    parser.add_argument('--vision_dset', action = 'store_true')
    parser.add_argument('--task', default='regression', type=str,choices = ['binary','multiclass','regression'])
    parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])

    parser.add_argument('--optimizer', default='AdamW', type=str,choices = ['AdamW','Adam','SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str,choices = ['cosine','linear'])

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=1, type=int)# ================================== 100
    parser.add_argument('--batchsize', default=256, type=int)

    parser.add_argument('--pretrain', action = 'store_true')
    parser.add_argument('--pretrain_epochs', default=1, type=int)# ================================== 50
    parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
    parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)

    parser.add_argument('--ssl_avail_y', default= 0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

    parser.add_argument('--results_path', type=Path, default=Path('../results'), help='result dir name')
    
    args = parser.parse_args()

    if args.task == 'regression':
        args.dtask = 'reg'
    else:
        args.dtask = 'clf'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")
    
    if same_reference_path!=None:
        args.save_path = args.results_path / same_reference_path
    else:
        args.save_path = args.results_path / time.strftime("%Y%m%d-%H%M%S")
        create_exp_dir(args.save_path, scripts_to_save=glob.glob('*.py'))

    preprocessing = Preprocessing()

    train_df, con_idxs, continuous, cat_idxs, categories, categorical_dims = preprocessing.get_train_data()# categories: カテゴリ列名, categorical_dims: カテゴリの次元
    categorical_dims = np.append(np.array([1]),np.array(categorical_dims)).astype(int)#Appending 1 for CLS token, this is later used to generate embeddings.
    test_df = preprocessing.get_test_data()

    train_X = train_df.drop("pm25_mid", axis=1)
    temp = train_X.drop(["id", "City"], axis=1).fillna("MissingValue")
    nan_mask = np.array(temp.ne("MissingValue").astype(int))
    del temp
    train_y = pd.DataFrame(np.array(train_df["pm25_mid"]).reshape(-1,1), columns=['pm25_mid']).reset_index(drop=True)
    groups = train_df["City"]

    if args.attentiontype != 'col':
        args.transformer_depth = 1
        args.attention_heads = min(4,args.attention_heads)
        args.attention_dropout = 0.8
        args.embedding_size = min(32,args.embedding_size)
        args.ff_dropout = 0.8

    if args.task == 'regression':
        y_dim = 1
    else:
        y_dim = len(np.unique(train_y[:,0]))

    # モデルの定義
    saint = SAINT_wrap(args, categorical_dims, cat_idxs, con_idxs, y_dim, device)
    saint.pretrain(train_X, nan_mask, train_y)

    # Cross Validation
    group_kf = GroupKFold(n_splits=num_folds)
    for random_seed, (train_index, val_index) in enumerate(group_kf.split(train_X, train_y, groups)):
        print("\n[CHECK POINT]: START ", random_seed+1, "\"fold\"\n")
        print("train city: ", train_X.iloc[train_index]["City"].unique())
        X_train = train_X.iloc[train_index].drop(["id", "City"], axis=1).reset_index(drop=True)
        mask_train = nan_mask[train_index]
        if X_train.shape != mask_train.shape:
            raise'Shape of data not same as that of nan mask!'
        y_train = pd.DataFrame(np.array(train_y.iloc[train_index]).reshape(-1,1), columns=['pm25_mid']).reset_index(drop=True)

        print("val city: ", train_X.iloc[val_index]["City"].unique())
        X_valid = train_X.iloc[val_index].drop(["id", "City"], axis=1).reset_index(drop=True)
        mask_valid = nan_mask[val_index]
        if X_valid.shape != mask_valid.shape:
            raise'Shape of data not same as that of nan mask!'
        y_valid = pd.DataFrame(np.array(train_y.iloc[val_index]).reshape(-1,1), columns=['pm25_mid']).reset_index(drop=True)

        train_mean, train_std = np.array(X_train.iloc[:,con_idxs],dtype=np.float32).mean(0), np.array(X_train.iloc[:,con_idxs],dtype=np.float32).std(0)
        train_std = np.where(train_std < 1e-6, 1e-6, train_std)
        continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

        train_ds = DataSetCatCon(X_train, mask_train, y_train, cat_idxs, con_idxs, args.dtask, continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True,num_workers=os.cpu_count())
        valid_ds = DataSetCatCon(X_valid, mask_valid, y_valid, cat_idxs, con_idxs, args.dtask, continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=args.batchsize, shuffle=False,num_workers=os.cpu_count())
        
        saint.train(trainloader, validloader, random_seed=random_seed)
    
        print("\n[CHECK POINT]: END \"", random_seed+1, "fold\"\n")

    print(f"\n[CHECK POINT]: END \"whole learning of SAINT.\" mean_of_train_time={np.mean(saint.train_times)}\n")

    return args.save_path, test_df, con_idxs, continuous, cat_idxs, categories, categorical_dims

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