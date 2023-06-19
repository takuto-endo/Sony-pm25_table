
"""
SAINTのモデル定義
"""

import numpy as np
np.random.seed(seed=0)
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error
import torch
torch.manual_seed(0)
from torch import nn
import torch.optim as optim
import time

from SAINT_models import SAINT

from SAINT_augmentations import embed_data_mask
from SAINT_augmentations import add_noise
from SAINT_augmentations import mixup_data
from SAINT_utils import get_scheduler, count_parameters, classification_scores, mean_sq_error

from SAINT_Preprocessing import *

import copy
import pickle

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


class SAINT_wrap(object):
    """docstring for LightGBM"""
    def __init__(self, args, cat_dims, cat_idxs, con_idxs, y_dim, device):
        super(SAINT_wrap, self).__init__()

        # param of tabnet
        self.num_model = 1
        self.saint_params = dict(categories = tuple(cat_dims), 
                                num_continuous = len(con_idxs),                
                                dim = args.embedding_size,                           
                                dim_out = 1,                       
                                depth = args.transformer_depth,                       
                                heads = args.attention_heads,                         
                                attn_dropout = args.attention_dropout,             
                                ff_dropout = args.ff_dropout,                  
                                mlp_hidden_mults = (4, 2),       
                                cont_embeddings = args.cont_embeddings,
                                attentiontype = args.attentiontype,
                                final_mlp_style = args.final_mlp_style,
                                y_dim = y_dim)

        # Box for strage
        self.device = device
        self.models = []
        self.args = args
        self.args.importance_path = self.args.save_path / "saint_importance"
        self.args.importance_path.mkdir(parents=True, exist_ok=True)
        self.args.model_path = self.args.save_path / "saint_model"
        self.args.model_path.mkdir(parents=True, exist_ok=True)
        self.valid_scores = []
        self.train_times = []

        self.cat_dims = cat_dims
        self.cat_idxs = cat_idxs
        self.con_idxs = con_idxs
        self.y_dim = y_dim

        self.pretrainer = None

    def SAINT_pretrain(self, model, X_train, mask, y_train):# args >> self.args

        cat_idxs = self.cat_idxs
        con_idxs = self.con_idxs
        device = self.device

        train_mean, train_std = np.array(X_train.iloc[:,con_idxs],dtype=np.float32).mean(0), np.array(X_train.iloc[:,con_idxs],dtype=np.float32).std(0)
        train_std = np.where(train_std < 1e-6, 1e-6, train_std)
        continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32) 

        train_ds = DataSetCatCon(X_train, mask, y_train, cat_idxs, con_idxs, self.args.dtask, continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=self.args.batchsize, shuffle=True,num_workers=os.cpu_count())

        optimizer = optim.AdamW(model.parameters(),lr=0.0001)
        pt_aug_dict = {
            'noise_type' : self.args.pt_aug,
            'lambda' : self.args.pt_aug_lam
        }
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        print("Pretraining begins!")
        print(gpulife())

        for epoch in range(self.args.pretrain_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                x_categ, x_cont, _ ,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
                
                # embed_data_mask function is used to embed both categorical and continuous data.
                if 'cutmix' in self.args.pt_aug:# === embedding前のaugmentation"cutmix"
                    x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                    _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model,self.args.vision_dset)
                else:# === cutmixは無し
                    _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,self.args.vision_dset)
                _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,self.args.vision_dset)# contrasive, self_supervisedの時に正解として扱う
                
                if 'mixup' in self.args.pt_aug:# === embedding後のaugmentation"cutmix"
                    x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=self.args.mixup_lam)

                loss = 0
                if 'contrastive' in self.args.pt_tasks:# contrasive self_supervised
                    aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)# SAINTに通す
                    aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)# SAINTに通す
                    aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                    aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                    if self.args.pt_projhead_style == 'diff':# 比較のためのprojection
                        aug_features_1 = model.pt_mlp(aug_features_1)
                        aug_features_2 = model.pt_mlp2(aug_features_2)
                    elif self.args.pt_projhead_style == 'same':# 比較のためのprojection,両方とも同じmlpでprojection
                        aug_features_1 = model.pt_mlp(aug_features_1)
                        aug_features_2 = model.pt_mlp(aug_features_2)
                    else:
                        print('Not using projection head')
                    logits_per_aug1 = aug_features_1 @ aug_features_2.t()/self.args.nce_temp
                    logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/self.args.nce_temp
                    targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                    loss_1 = criterion1(logits_per_aug1, targets)
                    loss_2 = criterion1(logits_per_aug2, targets)
                    loss   = self.args.lam0*(loss_1 + loss_2)/2
                elif 'contrastive_sim' in self.args.pt_tasks:# simpleなcontrasive, 距離を測るのは片方向だけだし, profection head はdiffで固定
                    aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                    aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                    aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                    aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                    c1 = aug_features_1 @ aug_features_2.t()
                    loss+= self.args.lam1*torch.diagonal(-1*c1).add_(1).pow_(2).sum()
                if 'denoising' in self.args.pt_tasks:# denoising self_supervised: 入力と比較する
                    cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                    # if con_outs.shape(-1) != 0:
                    # import ipdb; ipdb.set_trace()
                    if len(con_outs) > 0:
                        con_outs =  torch.cat(con_outs,dim=1)
                        l2 = criterion2(con_outs, x_cont)
                    else:
                        l2 = 0
                    l1 = 0
                    # import ipdb; ipdb.set_trace()
                    n_cat = x_categ.shape[-1]
                    for j in range(1,n_cat):
                        l1+= criterion1(cat_outs[j],x_categ[:,j])
                    loss += self.args.lam2*l1 + self.args.lam3*l2    
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch: {epoch}, Running Loss: {running_loss}')
            print(gpulife())

        print('END OF PRETRAINING!')
        print(gpulife())
        return model


    def pretrain(self, X_train, mask, y_train):
        self.pretrainer = SAINT(**self.saint_params)
        self.pretrainer.to(self.device)

        print("\n[CHECK POINT]: START \"pretrain saint.\"\n")
        print(gpulife())
        train_start = time.time()
        self.pretrainer = self.SAINT_pretrain(self.pretrainer, X_train.drop(["id", "City"], axis=1), mask, y_train)
        train_time = time.time() - train_start
        print(f"\n[CHECK POINT]: END \"pretrain saint.\" train_time={train_time}\n")
        print(gpulife())

    def train(self, trainloader, validloader, print_importance=True, random_seed=0):

        device = self.device

        if self.y_dim == 2 and self.args.task == 'binary':
            # opt.task = 'binary'
            criterion = nn.CrossEntropyLoss().to(device)
        elif self.y_dim > 2 and  self.args.task == 'multiclass':
            # opt.task = 'multiclass'
            criterion = nn.CrossEntropyLoss().to(device)
        elif self.args.task == 'regression':
            criterion = nn.MSELoss().to(device)
        else:
            raise'case not written yet'

        for s in range(self.num_model):
            torch.manual_seed(random_seed*self.num_model+s)
            torch.manual_seed(random_seed*self.num_model+s)

            print("\n[CHECK POINT]: START \"train SAINT.\"\n")
            print(gpulife())
            train_start = time.time()

            if self.pretrainer != None:
                model = copy.deepcopy(self.pretrainer)
            else:
                model = SAINT(**self.saint_params)
            model.to(device)

            ## Choosing the optimizer

            if self.args.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=self.args.lr,
                                      momentum=0.9, weight_decay=5e-4)
                scheduler = get_scheduler(self.args, optimizer)
            elif self.args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(),lr=self.args.lr)
            elif self.args.optimizer == 'AdamW':
                optimizer = optim.AdamW(model.parameters(),lr=self.args.lr)
            best_valid_auroc = 0
            best_valid_accuracy = 0
            best_test_auroc = 0
            best_test_accuracy = 0
            best_valid_rmse = 100000
            print('Training begins now.')

            for epoch in range(self.args.epochs):
                model.train()
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    optimizer.zero_grad()
                    # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
                    x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)

                    # We are converting the data to embeddings in the next step
                    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,self.args.vision_dset)           
                    reps = model.transformer(x_categ_enc, x_cont_enc)
                    # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
                    y_reps = reps[:,0,:]
                    
                    y_outs = model.mlpfory(y_reps)
                    if self.args.task == 'regression':
                        loss = criterion(y_outs,y_gts) 
                    else:
                        loss = criterion(y_outs,y_gts.squeeze()) 
                    loss.backward()
                    optimizer.step()
                    if self.args.optimizer == 'SGD':
                        scheduler.step()
                    running_loss += loss.item()

                if epoch%5==0:
                    model.eval()
                    with torch.no_grad():
                        valid_rmse = mean_sq_error(model, validloader, device, self.args.vision_dset)     
                        print('[EPOCH %d] VALID RMSE: %.3f' %
                            (epoch + 1, valid_rmse ))  
                        if valid_rmse < best_valid_rmse:
                            best_valid_rmse = valid_rmse
                            best_epoch = epoch

                            saved_filepath = f'{self.args.model_path}/SAINT{random_seed*self.num_model+s}.pth'
                            torch.save(model.state_dict(),saved_filepath)

                            with open(f'{self.args.model_path}/SAINT_params{random_seed*self.num_model+s}.json', 'wb') as fp:
                                pickle.dump(self.saint_params, fp)

                            print(f'model saved. in Epoch{epoch} >> rmse:{valid_rmse}')
                    model.train()

            train_time = time.time() - train_start
            print(f"\n[CHECK POINT]: END \"train SAINT.\" train_time={train_time}\n")
            print(gpulife())
            self.train_times.append(train_time)


            self.models.append(model)
            self.valid_scores.append(best_valid_rmse)

            print(f"SAINT valid rmse: {best_valid_rmse} in Epoch{best_epoch}")
            print("valid rmse scores: ", self.valid_scores)

            if print_importance:
                pass

            if self.args.importance_path != None:
                pass

            print("saved_filepath: ", saved_filepath)


if __name__ == "__main__":
    pass