
import pandas as pd

import time
import glob
from pathlib import Path
import argparse
import os
import shutil

import LightGBM_Predicting
import TabNet_Predicting
import SAINT_Predicting

def main():

    parser = argparse.ArgumentParser(description='sony_pm25')
    parser.add_argument('--save_same_dir', type=bool, default=True, help='save same directory')
    parser.add_argument('--results_path', type=Path, default=Path('../results'), help='result dir name')
    parser.add_argument('--reference_path', type=Path, default=None, help='result dir name')# 指定された場合はtrainingなし
    args = parser.parse_args()

    if args.save_same_dir:

        if args.reference_path!=None:
            print("reference_path: ", args.reference_path)
            reference_path = str(args.reference_path)
            args.save_path = args.results_path / args.reference_path
            load_model = True
        else:
            reference_path = time.strftime("%Y%m%d-%H%M%S")
            args.save_path = args.results_path / reference_path
            create_exp_dir(args.save_path, scripts_to_save=glob.glob('*.py'))
            load_model = False

        tabnet_submit_path = TabNet_Predicting.predicting(same_reference_path=reference_path, load_model=load_model)
        saint_submit_path = SAINT_Predicting.predicting(same_reference_path=reference_path, load_model=load_model)
        lgbm_submit_path = LightGBM_Predicting.predicting(same_reference_path=reference_path, load_model=load_model)

        lgbm_submit = pd.read_csv(lgbm_submit_path, header=None)
        tabnet_submit = pd.read_csv(tabnet_submit_path, header=None)
        saint_submit = pd.read_csv(saint_submit_path, header=None)

        ensemble_submit = pd.read_csv("../data/submit_sample.csv", header=None)
        ensemble_submit[1] = lgbm_submit[1]*0.5 + tabnet_submit[1]*0.3 + saint_submit[1]*0.2
        ensemble_submit.to_csv(str(args.save_path)+"/ensemble_submit_file_"+reference_path+".csv",index=False, header=False)
        print("[CHECK POINT]: END save submission file >> [", str(args.save_path)+"/ensemble_submit_file_"+reference_path+".csv ]")

    else:
        _ = LightGBM_Predicting.predicting()
        _ = TabNet_Predicting.predicting()
        _ = SAINT_Predicting.predicting()



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
    
    main()
