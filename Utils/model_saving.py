'''
@author: weiboshu2

@file: model_saving.py

@time: 2021/9/5 9:59

@purpose: this file works for saving model

'''

import os
import numpy as np
import torch
import logging


class CC_Model_Saver():
    def __init__(self, path):
        save_path = path.split('/')
        prefix = '/'.join(save_path[:-1])

        suffix1 = save_path[-1].split('.')[0] + '_loss.pth'
        suffix2 = save_path[-1].split('.')[0] + '_mae.pth'
        suffix3 = save_path[-1].split('.')[0] + '_mse.pth'

        directory = 'Model/model_pretrain/' + prefix
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model_save_path1 = '/'.join((directory, suffix1))
        self.model_save_path2 = '/'.join((directory, suffix2))
        self.model_save_path3 = '/'.join((directory, suffix3))

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_loss = np.inf

    def save(self, epoch, model, mae=None, mse=None, loss=None):
        try:
            if loss < self.best_loss:
                self.best_loss = loss
                logging.debug("save best loss {:.2f} model epoch {}".format(self.best_loss, epoch))
                model_state_dic = model.state_dict()
                torch.save(model_state_dic, self.model_save_path1)
        except TypeError as e:
            if 'NoneType' in str(e):
                pass

        try:
            if mae < self.best_mae:
                self.best_mae = mae
                logging.debug("save best mae {:.2f} model epoch {}".format(self.best_mae, epoch))
                model_state_dic = model.state_dict()
                torch.save(model_state_dic, self.model_save_path2)
        except TypeError as e:
            if 'NoneType' in str(e):
                pass

        try:
            if mse < self.best_mse:
                self.best_mse = mse
                logging.debug("save best mse {:.2f} model epoch {}".format(self.best_mse, epoch))
                model_state_dic = model.state_dict()
                torch.save(model_state_dic, self.model_save_path3)
        except TypeError as e:
            if 'NoneType' in str(e):
                pass



