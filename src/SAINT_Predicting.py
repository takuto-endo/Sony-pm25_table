
""" 公式 disctiption
Preprocessingで作成した評価用データ及び
Learningで作成した学習済みモデルを読み込み、予測結果を出力するモジュール。
"""

from SAINT_Preprocessing import *
from SAINT_Learning import *

import time
import glob
from pathlib import Path
import argparse
import os
import shutil
import sys

from torch.utils.data import DataLoader

import torch
from SAINT_models import SAINT
import pickle

def predicting(same_reference_path=None, load_model=False):

    parser = argparse.ArgumentParser(description='sony_pm25')
    parser.add_argument('--reference_path', type=Path, default=None, help='result dir name')
    parser.add_argument('--batchsize', default=256, type=int)
    args = parser.parse_args()

    if load_model:
        args.reference_path = same_reference_path

    if args.reference_path != None:# 参照するpathがある場合, 既にlearningが終わっている
        print("reference_path: ", args.reference_path)
        preprocessing = Preprocessing()
        train_df, con_idxs, continuous, cat_idxs, categories, categorical_dims = preprocessing.get_train_data()# categories: カテゴリ列名, categorical_dims: カテゴリの次元
        categorical_dims = np.append(np.array([1]),np.array(categorical_dims)).astype(int)#Appending 1 for CLS token, this is later used to generate embeddings.
        test_df = preprocessing.get_test_data()
        save_path = "../results/" + str(args.reference_path)
        reference_path = str(args.reference_path)
    else:
        save_path, test_df, con_idxs, continuous, cat_idxs, categories, categorical_dims  = learning(same_reference_path=same_reference_path)
        save_path = str(save_path)
        reference_path = save_path.split('/')[-1]

    X_test = test_df.drop("id", axis=1)
    temp = X_test.fillna("MissingValue")
    mask_test = np.array(temp.ne("MissingValue").astype(int))
    del temp

    test_mean, test_std = np.array(X_test.iloc[:,con_idxs],dtype=np.float32).mean(0), np.array(X_test.iloc[:,con_idxs],dtype=np.float32).std(0)
    test_std = np.where(test_std < 1e-6, 1e-6, test_std)
    continuous_mean_std = np.array([test_mean,test_std]).astype(np.float32) 

    print(X_test.shape)
    print(mask_test.shape)

    test_ds = TestDataSetCatCon(X_test, mask_test, cat_idxs, con_idxs, 'reg', continuous_mean_std)
    testloader = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=os.cpu_count())

    submit_df = pd.read_csv("../data/submit_sample.csv", header=None)
    preds = []

    # model load saint
    model_path = '/saint_model/*.pth'
    fnames = glob.glob(save_path+model_path)
    print(save_path+model_path)
    print("fnames: ", fnames)
    param_path = '/saint_model/*.json'
    param_fnames = glob.glob(save_path+param_path)
    print(save_path+param_path)
    print("param_fnames: ", param_fnames)


    inference_times = []

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for param_fname, fname in zip(param_fnames, fnames):
        print("param_fname: ", param_fname)
        print("fname: ", fname)

        with open(param_fname, 'rb') as fp:
            params = pickle.load(fp)
            print('params: \n', params)

        model = SAINT(**params)
        model = model.to(device)
        model.load_state_dict(torch.load(fname))
        # from torchsummary import summary
        # summary(model, input_size=([2,32], [50,32]))

        print("[CHECK POINT]: START inference of ", fname)
        inference_start = time.time()

        pred = []
        print("length of testloader: ", len(testloader))
        for i, data in enumerate(testloader):
            # print("data", data)
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model, 'reg') 
            reps = model.transformer(x_categ_enc, x_cont_enc) 
            # print("reps: ", reps.shape)
            y_reps = reps[:,0,:]
            # print("y_reps: ", y_reps.shape)
            y_outs = model.mlpfory(y_reps)
            # print("y_outs: ", y_outs.shape)

            pred.append(y_outs.to('cpu').detach().numpy().copy())

        pred = np.concatenate(pred, 0)
        print("pred shape: ", pred.shape)

        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        print(f"[CHECK POINT]: END inference of {fname} inference_time={inference_time} X_test_shape={X_test.shape}")

        preds.append(pred)

        
    print(f"\n[CHECK POINT]: END \"whole inference of tabnet.\" mean_of_inference_time={np.mean(inference_times)}\n")
    print(f"all inference_time={inference_times}")

    preds = np.mean(np.array(preds), axis=0)
    submit_df[1] = preds
    submit_df.to_csv(save_path+"/saint_submit_file_"+reference_path+".csv",index=False, header=False)
    print("[CHECK POINT]: END save submission file >> [", save_path+"/saint_submit_file_"+reference_path+".csv ]")

    return save_path+"/saint_submit_file_"+reference_path+".csv"

if __name__ == "__main__":
    _ = predicting()