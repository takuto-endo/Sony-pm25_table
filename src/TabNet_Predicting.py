
""" 公式 disctiption
Preprocessingで作成した評価用データ及び
Learningで作成した学習済みモデルを読み込み、予測結果を出力するモジュール。
"""

from TabNet_Preprocessing import *
from TabNet_Learning import *

import time
import glob
from pathlib import Path
import argparse
import os
import shutil
import sys

from pytorch_tabnet.tab_model import TabNetRegressor


def predicting(same_reference_path=None, load_model=False):

    parser = argparse.ArgumentParser(description='sony_pm25')
    parser.add_argument('--reference_path', type=Path, default=None, help='result dir name')
    args = parser.parse_args()

    if load_model:
        args.reference_path = same_reference_path

    if args.reference_path != None:
        print("reference_path: ", args.reference_path)
        preprocessing = Preprocessing()
        train_df, categories, categorical_dims = preprocessing.get_train_data()
        test_df = preprocessing.get_test_data()
        save_path = "../results/" + str(args.reference_path)
        reference_path = str(args.reference_path)
    else:
        save_path, test_df = learning(same_reference_path=same_reference_path)
        save_path = str(save_path)
        reference_path = save_path.split('/')[-1]

    X_test = test_df.drop("id", axis=1)
    submit_df = pd.read_csv("../data/submit_sample.csv", header=None)
    preds = []

    # model load lightgbm
    fnames = glob.glob(save_path+"/tabnet_model/*.zip")
    inference_times = []
    model_sizes = []
    for i, fname in enumerate(fnames):
        loaded_clf = TabNetRegressor()
        loaded_clf.load_model(fname)
        print("[CHECK POINT]: START inference of ", fname)
        inference_start = time.time()
        pred = loaded_clf.predict(X_test.values)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        print(f"[CHECK POINT]: END inference of {fname} inference_time={inference_time} X_test_shape={X_test.shape}")
        model_size = sys.getsizeof(loaded_clf)
        model_sizes.append(model_size)
        print(f"Model size: {model_size} byte.")
        preds.append(pred)
    print(f"\n[CHECK POINT]: END \"whole inference of tabnet.\" mean_of_inference_time={np.mean(inference_times)}\n")
    print(f"all inference_time={inference_times}")
    print(f"mean of model size : {np.mean(model_sizes)} byte.")
    print(f"all model size : {model_sizes}")

    preds = np.mean(np.array(preds), axis=0)
    submit_df[1] = preds
    submit_df.to_csv(save_path+"/tabnet_submit_file_"+reference_path+".csv",index=False, header=False)
    print("[CHECK POINT]: END save submission file >> [", save_path+"/tabnet_submit_file_"+reference_path+".csv ]")

    return save_path+"/tabnet_submit_file_"+reference_path+".csv"

if __name__ == "__main__":
    _ = predicting()