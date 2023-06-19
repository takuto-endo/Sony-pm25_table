from LightGBM_Preprocessing import *
from LightGBM import *

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

import optuna

# 最適化
def objective(trial):

    preprocessing = Preprocessing()
    train_df, categories = preprocessing.get_train_data()
    test_df = preprocessing.get_test_data()

    train_X = train_df.drop("pm25_mid", axis=1)
    train_y = train_df["pm25_mid"]
    groups = train_df["City"]

    params = {
        "objective":"regression",
        "metric":"rmse",
        "learning_rate":0.05,
        "random_seed":0,
        "verbosity": 0,
        "force_col_wise":True,

        "max_depth" : trial.suggest_int("max_depth",3,12),
        "num_leaves":trial.suggest_int("num_leaves",4,25),
        "max_bin":trial.suggest_int("max_bin",50,200),
        "bagging_fraction":trial.suggest_uniform("bagging_fraction",0.2,0.9),
        "bagging_freq":trial.suggest_int("bagging_freq",1,10),
        "feature_fraction":trial.suggest_uniform("feature_fraction",0.2,0.9),
        "min_data_in_leaf":trial.suggest_int("min_data_in_leaf",2,16),
        "min_sum_hessian_in_leaf":trial.suggest_int("min_sum_hessian_in_leaf",1,10)
    }

    # Cross Validation
    scores = []
    num_folds = 5
    group_kf = GroupKFold(n_splits=num_folds)
    for random_seed, (train_index, val_index) in enumerate(group_kf.split(train_X, train_y, groups)):
        print("\n[CHECK POINT]: START ", random_seed+1, "\"fold\"\n")
        print("train city: ", train_X.iloc[train_index]["City"].unique())
        X_train = train_X.iloc[train_index].drop("City", axis=1)
        y_train = train_y.iloc[train_index]
        print("val city: ", train_X.iloc[val_index]["City"].unique())
        X_valid = train_X.iloc[val_index].drop("City", axis=1)
        y_valid = train_y.iloc[val_index]

        lgb_train = lgb.Dataset(X_train.drop("id", axis=1), y_train)
        lgb_eval = lgb.Dataset(X_valid.drop("id", axis=1), y_valid)

        params["random_seed"] = random_seed
        
        model = lgb.train(params,
                        lgb_train,
                        valid_sets = [lgb_train, lgb_eval],
                        categorical_feature = ["Country"],
                        valid_names = ["Train", "Eval"],
                        num_boost_round = 10000,
                        early_stopping_rounds = 50,
                        verbose_eval = 100)

        y_pred = model.predict(X_valid.drop("id", axis=1), num_iteration=model.best_iteration)
        score = np.sqrt(mean_squared_error(y_valid, y_pred))
        scores.append(score)
    score = np.mean(scores)
    print("mean rmse: ",score)
    return score


if __name__ == "__main__":
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    study.optimize(objective, n_trials=50)
    print(study.best_params)


