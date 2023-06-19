
"""
TabNetのモデル定義
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error
import torch
import time

import subprocess
import shlex
def gpulife():
    """
    Returns GPU usage information in string.

    Parameters
    ----------
    None

    Returns
    -------
    msg: str
    """

    def _gpuinfo():
        command = 'nvidia-smi -q -d MEMORY | sed -n "/FB Memory Usage/,/Free/p" | sed -e "1d" -e "4d" -e "s/ MiB//g" | cut -d ":" -f 2 | cut -c2-'
        commands = [shlex.split(part) for part in command.split(' | ')]
        for i, cmd in enumerate(commands):
            if i==0:
                res = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            else:
                res = subprocess.Popen(cmd, stdin=res.stdout, stdout=subprocess.PIPE)
        return tuple(map(int, res.communicate()[0].decode('utf-8').strip().split('\n')))

    total, used = _gpuinfo()
    percent = int(used / total * 100)
    msg = 'GPU RAM Usage: {} {}/{} MiB ({:.1f}%)'.format('|' * (percent // 5) + '.' * (20 - percent // 5), used, total, used/total*100)
    return msg

class TabNet(object):
    """docstring for LightGBM"""
    def __init__(self, args):
        super(TabNet, self).__init__()

        # param of tabnet
        self.num_model = 1
        """ [[[default]]]
        self.tabnet_params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                                            n_independent=2, n_shared=2,
                                            seed=0, lambda_sparse=1e-3, 
                                            optimizer_fn=torch.optim.Adam, 
                                            optimizer_params=dict(lr=2e-2),
                                            mask_type="entmax",
                                            scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9,),
                                            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                            verbose=5)"""
        self.tabnet_params = dict(n_d=48, n_a=48, n_steps=6, gamma=2.0,
                    n_independent=2, n_shared=1,
                    seed=0, lambda_sparse=1.3096774493777606e-06, 
                    optimizer_fn=torch.optim.Adam, 
                    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                    mask_type="entmax",
                    scheduler_params=dict(mode="min", patience=10, min_lr=1e-5, factor=0.9,),
                    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    verbose=5)
        """[[[over memory]]]
        self.tabnet_params = dict(n_d=48, n_a=48, n_steps=6, gamma=1.6,
                                            n_independent=2, n_shared=3,
                                            seed=0, lambda_sparse=1.113512318625358e-05, 
                                            optimizer_fn=torch.optim.Adam, 
                                            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                                            mask_type="sparsemax",
                                            scheduler_params=dict(mode="min", patience=10, min_lr=1e-5, factor=0.9,),
                                            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                            verbose=5)"""

        # Box for strage
        self.models = []
        self.args = args
        self.args.importance_path = self.args.save_path / "tabnet_importance"
        self.args.importance_path.mkdir(parents=True, exist_ok=True)
        self.args.model_path = self.args.save_path / "tabnet_model"
        self.args.model_path.mkdir(parents=True, exist_ok=True)
        self.valid_scores = []
        self.train_times = []

    def pretrain(self, train_X):
        self.pretrainer = TabNetPretrainer(**self.tabnet_params)
        print("\n[CHECK POINT]: START \"pretrain tabnet.\"\n")
        print(gpulife())
        train_start = time.time()
        self.pretrainer.fit(
            X_train=train_X.drop(['id', 'City'],axis=1).values,
            eval_set=[train_X.drop(['id', 'City'],axis=1).values],
            max_epochs=1,# ================================== 200
            patience=20, batch_size=256, virtual_batch_size=128,
            num_workers=os.cpu_count(), drop_last=True)
        train_time = time.time() - train_start
        print(f"\n[CHECK POINT]: END \"pretrain tabnet.\" train_time={train_time}\n")
        print(gpulife())

    def train(self, X_train, y_train, X_valid, y_valid, categories=None, categorical_dims=None, print_importance=True, random_seed=0):

        print(X_train)
        print(X_valid)

        unused_feat = ['id']
        target = "pm25_mid"
        features = [ col for col in X_train.columns if col not in unused_feat] 
        cat_idxs = [ i for i, f in enumerate(features) if f in categories]
        cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categories]
        # define your embedding sizes : here just a random choice
        cat_emb_dim = [10]

        for s in range(self.num_model):
            self.tabnet_params["seed"] = random_seed*self.num_model+s

            clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

            print("\n[CHECK POINT]: START \"train tabnet.\"\n")
            print(gpulife())
            train_start = time.time()
            clf.fit(
                X_train=X_train.values, y_train=y_train.values,
                eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
                eval_name=['train', 'valid'],
                eval_metric=['rmse', 'rmsle', 'mae', 'mse'],
                max_epochs=1,# ================================== 200
                patience=50,
                batch_size=1024, virtual_batch_size=128,
                num_workers=os.cpu_count(),
                drop_last=False,
                from_unsupervised=self.pretrainer,
                # augmentations=aug, #aug
            ) 
            train_time = time.time() - train_start
            print(f"\n[CHECK POINT]: END \"train tabnet.\" train_time={train_time}\n")
            print(gpulife())
            self.train_times.append(train_time)

            self.models.append(clf)

            y_pred = clf.predict(X_valid.values)
            score = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_valid))
            self.valid_scores.append(score)

            print("TabNet valid rmse: ", score)
            print("valid rmse scores: ", self.valid_scores)

            if print_importance:
                print(" Feature Importance ")
                importance = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
                print(importance.sort_values('importance', ascending=False))

            if self.args.importance_path != None:
                explain_matrix, masks = clf.explain(X_valid.values)
                print(len(masks))
                fig, axs = plt.subplots(1, 3, figsize=(30,30))
                for i in range(3):
                    axs[i].imshow(masks[i][:50])
                    axs[i].set_title(f"mask {i}")
                plt.savefig(str(self.args.importance_path)+'/impoprtance_TabNet'+str(random_seed*self.num_model+s))

            # save tabnet model
            saving_path_name = str(self.args.model_path)+'/TabNet'+str(random_seed*self.num_model+s)
            saved_filepath = clf.save_model(saving_path_name)
            print(saved_filepath)


if __name__ == "__main__":
    pass