from SAINT_Preprocessing import *

import numpy as np
np.random.seed(seed=0)
import pandas as pd
from sklearn.model_selection import KFold
import torch
import os

import optuna

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

# 最適化
def objective(trial):

    preprocessing = Preprocessing()
    train_df, categories, categorical_dims = preprocessing.get_train_data()
    test_df = preprocessing.get_test_data()

    train_X = train_df.drop("pm25_mid", axis=1)
    train_y = train_df["pm25_mid"]
    groups = train_df["City"]

    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_da = trial.suggest_int("n_da", 8, 64, step=8)# 8, 64, step=8
    n_steps = trial.suggest_int("n_steps", 1, 6, step=1)# 1, 10, step=3
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)# 1.0, 2.0, step=0.2
    n_shared = trial.suggest_int("n_shared", 1, 3)# 1, 3
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)# 1e-6, 1e-3, log=True

    tabnet_params = dict(
        n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
        lambda_sparse=lambda_sparse, mask_type=mask_type, n_shared=n_shared,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_params=dict(
            mode="min", patience=10,
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0,
        ),
        seed=0,
    )


    unused_feat = ['id', 'City']
    target = "pm25_mid"
    features = [ col for col in train_X.columns if col not in unused_feat] 
    cat_idxs = [ i for i, f in enumerate(features) if f in categories]
    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categories]
    # define your embedding sizes : here just a random choice
    cat_emb_dim = [10]
    print(tabnet_params)

    pretrainer = TabNetPretrainer(**tabnet_params, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)
    pretrainer.fit(
        X_train=train_X.drop(['id', 'City'],axis=1).values,
        eval_set=[train_X.drop(['id', 'City'],axis=1).values],
        max_epochs=200,#200
        patience=20, batch_size=512, virtual_batch_size=128,# 1024
        num_workers=os.cpu_count(), drop_last=True)

    # Cross Validation
    scores = []
    num_folds = 3# 5
    group_kf = GroupKFold(n_splits=num_folds)
    for random_seed, (train_index, val_index) in enumerate(group_kf.split(train_X, train_y, groups)):

        print("\n[CHECK POINT]: START ", random_seed+1, "\"fold\"\n")
        # print("train city: ", train_X.iloc[train_index]["City"].unique())
        X_train = train_X.iloc[train_index].drop(["id", "City"], axis=1).reset_index(drop=True)
        y_train = pd.DataFrame(np.array(train_y.iloc[train_index]).reshape(-1,1), columns=['pm25_mid']).reset_index(drop=True)
        # print("val city: ", train_X.iloc[val_index]["City"].unique())
        X_valid = train_X.iloc[val_index].drop(["id", "City"], axis=1).reset_index(drop=True)
        y_valid = pd.DataFrame(np.array(train_y.iloc[val_index]).reshape(-1,1), columns=['pm25_mid']).reset_index(drop=True)

        """unused_feat = ['id']
            target = "pm25_mid"
            features = [ col for col in X_train.columns if col not in unused_feat] 
            cat_idxs = [ i for i, f in enumerate(features) if f in categories]
            cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categories]
            # define your embedding sizes : here just a random choice
            cat_emb_dim = [10]"""

        clf = TabNetRegressor(**tabnet_params, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)
        clf.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
            eval_name=['train', 'valid'],
            eval_metric=['rmse', 'rmsle', 'mae', 'mse'],
            max_epochs=200,# 100
            patience=30,# 50
            batch_size=512, virtual_batch_size=128,
            num_workers=os.cpu_count(),
            drop_last=False,
            from_unsupervised=pretrainer,
            # augmentations=aug, #aug
        ) 

        y_pred = clf.predict(X_valid.values)
        score = np.sqrt(mean_squared_error(y_valid, y_pred))
        scores.append(score)

        del clf
    del pretrainer
        
    score = np.mean(scores)
    print("mean rmse: ",score)
    return score


if __name__ == "__main__":
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    study.optimize(objective, n_trials=50)
    print(study.best_params)


