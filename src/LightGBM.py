
"""
LightGBMのモデル定義
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import time

class LightGBM(object):
    """docstring for LightGBM"""
    def __init__(self, args):
        super(LightGBM, self).__init__()

        # param of lightgbm
        self.num_model = 1
        self.num_boost_round = 10000
        self.learning_rate = 0.01
        """ [[[default]]]
        self.params = {
                        
                                    # Fixed Parameters
                                    "objective":"regression",
                                    "metric":"rmse",
                                    "learning_rate":self.learning_rate,
                                    "random_seed":0,
                                    "force_col_wise":True,
                                    # "device":"gpu",
                        
                                    # Variable Parameters
                                    'max_depth': 8,
                                    'num_leaves': 20, 
                                    'max_bin': 128, 
                                    'bagging_fraction': 0.7396187373513217, 
                                    'bagging_freq': 6, 
                                    'feature_fraction': 0.5259969150143866, 
                                    'min_data_in_leaf': 7, 
                                    'min_sum_hessian_in_leaf': 1
                                }"""
        self.params = {

            # Fixed Parameters
            "objective":"regression",
            "metric":"rmse",
            "learning_rate":self.learning_rate,
            "random_seed":0,
            "force_col_wise":True,
            # "device":"gpu",

            # Variable Parameters
            'max_depth': 8, 
            'num_leaves': 18, 
            'max_bin': 71, 
            'bagging_fraction': 0.7260694719946708, 
            'bagging_freq': 3, 
            'feature_fraction': 0.5635462770562231, 
            'min_data_in_leaf': 13, 
            'min_sum_hessian_in_leaf': 1
        }

        # Box for strage
        self.models = []
        self.args = args
        self.args.importance_path = self.args.save_path / "lgbm_importance"
        self.args.importance_path.mkdir(parents=True, exist_ok=True)
        self.args.model_path = self.args.save_path / "lgbm_model"
        self.args.model_path.mkdir(parents=True, exist_ok=True)
        self.valid_scores = []
        self.train_times = []

    def train(self, X_train, y_train, X_valid, y_valid, categories=None, print_importance=True, random_seed=0):

        print(X_train.drop("id", axis=1))
        print(X_valid.drop("id", axis=1))

        lgb_train = lgb.Dataset(X_train.drop("id", axis=1), y_train)
        lgb_eval = lgb.Dataset(X_valid.drop("id", axis=1), y_valid)

        for s in range(self.num_model):
            self.params["random_seed"] = random_seed*self.num_model+s
            # training model
            lgb_results = {}
            print("\n[CHECK POINT]: START \"train lightgbm.\"\n")
            train_start = time.time()
            model = lgb.train(self.params,
                            lgb_train,
                            valid_sets = [lgb_train, lgb_eval],
                            categorical_feature = categories,
                            valid_names = ["Train", "Eval"],
                            num_boost_round = self.num_boost_round,
                            early_stopping_rounds = 50,
                            verbose_eval = 50,
                            evals_result = lgb_results)
            train_time = time.time() - train_start
            print(f"\n[CHECK POINT]: END \"train lightgbm.\" train_time={train_time}\n")
            self.train_times.append(train_time)

            self.models.append(model)

            y_pred = model.predict(X_valid.drop("id", axis=1), num_iteration=model.best_iteration)
            score = np.sqrt(mean_squared_error(y_valid, y_pred))
            self.valid_scores.append(score)

            print("LightGBM valid rmse: ", score)
            print("valid rmse scores: ", self.valid_scores)

            if print_importance:
                print(" Feature Importance ")
                importance = pd.DataFrame(model.feature_importance(importance_type="gain"), index=X_train.drop("id", axis=1).columns, columns=['importance'])
                print(importance.sort_values('importance', ascending=False))

            if self.args.importance_path != None:
                lgb.plot_importance(model, importance_type="gain", max_num_features=100)
                plt.savefig(str(self.args.importance_path)+"/importance"+str(self.params["random_seed"])+".png")

            model.save_model(str(self.args.model_path)+'/LightGBM'+str(random_seed*self.num_model+s)+'.txt')


if __name__ == "__main__":
    pass