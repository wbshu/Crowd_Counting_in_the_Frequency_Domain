import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import logging

from Dataset.dataloader import Collate_F
from Dataset.preprocessor import Directory_path
import Dataset.dataloader as dl
from Loss import chfloss, chf_likelihood
from Utils.model_saving import CC_Model_Saver

logging.basicConfig(filename='logging', level=logging.DEBUG)


def optimizer_parser(params: dict, model: torch.nn.Module):
    if params['optimizer'].lower() == 'sgd':
        keys = params.keys()
        momentum = float(params['momentum']) if 'momentum' in keys else 0
        dampening = float(params['dampening']) if 'dampening' in keys else 0
        weight_decay = float(params['weight_decay']) if 'weight_decay' in keys else 0
        nesterov = bool(params['nesterov']) if 'nesterov' in keys else False
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=momentum, dampening=dampening,
                                    weight_decay=weight_decay, nesterov=nesterov)
    elif params['optimizer'].lower() == 'adam':
        keys = params.keys()
        lr = float(params['lr']) if 'lr' in keys else 0.001
        betas = tuple(params['betas']) if 'betas' in keys else (0.9, 0.999)
        eps = float(params['eps']) if 'eps' in keys else 1e-08
        weight_decay = float(params['weight_decay']) if 'weight_decay' in keys else 0
        amsgrad = bool(params['amsgrad']) if 'amsgrad' in keys else False
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                     amsgrad=amsgrad)
    return optimizer

def dataloader_parser(params: dict, dataset: str, category=('train', 'val')):
    '''
    :param params: it's a python dictionary, below is a template.

    a template:
    {'datahandler': 'chf_rcrop', 'batch_size':1, 'shorter_length_min':0, 'shorter_length_max':np.inf, 'is_gray':False}
    // except for 'datahandler', all have default values, the values in the template are their default value

    :param dataset:  SHTCA, SHTCB, QNRF,JHU++,NWPU, or your own dataset.
    :param category: the used sets in training process, usually set as default.
    :return:
    '''
    path_prefix, img_path_suffix, dot_path_suffix = Directory_path.prefix_suffix(dataset)

    keys = params.keys()
    batch_size = params['batch_size'] if 'batch_size' in keys else 1
    shorter_length_min = params['shorter_length_min'] if 'shorter_length_min' in keys else 0
    shorter_length_max = params['shorter_length_max'] if 'shorter_length_max' in keys else np.inf

    if params['datahandler'].lower() == 'chf_rcrop':
        input_img_size = params['img_size']
        chf_step = params['chf_step']
        chf_tik = params['chf_tik']
        bandwidth = params['bandwidth']
        device = params['device']
        is_gray = params['is_gray'] if 'is_gray' in keys else False

        datasets = {x: dl.ChfData_RCrop(path_prefix + x + img_path_suffix, path_prefix + x + dot_path_suffix, x,
                                        input_img_size, chf_step, chf_tik, min_size=shorter_length_min,
                                        max_size=shorter_length_max,
                                        bandwidth=bandwidth, device=device, is_gray=is_gray) for x in category}
        dataloaders = {x: DataLoader(datasets[x],
                                     collate_fn=(Collate_F.train_collate
                                                 if x.startswith('train') else default_collate),
                                     batch_size=batch_size if x.startswith('train') else 1,
                                     shuffle=(True if x.startswith('train') else False))
                       for x in category}
        return dataloaders

    elif params['datahandler'].lower() == 'hard_dish_chf_rcrop':
        input_img_size = params['img_size']
        chf_step = params['chf_step']
        chf_tik = params['chf_tik']
        bandwidth = params['bandwidth']
        device = params['device']
        is_gray = params['is_gray'] if 'is_gray' in keys else False
        datasets = {
            x: dl.ChfData_RCrop_Harddish_Load(path_prefix + x + img_path_suffix, path_prefix + x + dot_path_suffix,
                                              x, input_img_size, chf_step, chf_tik, min_size=shorter_length_min,
                                              max_size=shorter_length_max, bandwidth=bandwidth, device=device,
                                              is_gray=is_gray) for
            x in category}

        dataloaders = {x: DataLoader(datasets[x],
                                     collate_fn=(Collate_F.train_collate
                                                 if x.startswith('train') else default_collate),
                                     batch_size=batch_size if x.startswith('train') else 1,
                                     shuffle=(True if x.startswith('train') else False), num_workers=8, pin_memory=True
                                     )
                       for x in category}
        return dataloaders


class Crowd_couting_trainer():
    def __init__(self, logger, model, optimizer: dict, dataset: str, dataloader: dict, train_epoch: int = 1000,
                 best_model_save_path: str = 'best_model.pth', sample_interval=8, img_size=512,
                 set_category=('train', 'test')):
        # register global parameters used in train/val/test process
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # register global parameters
        self.sample_interval = sample_interval
        self.img_size = img_size

        # register parameters for dataloader
        dataloader['sample_interval'] = self.sample_interval
        dataloader['img_size'] = self.img_size
        dataloader['device'] = self.device
        self.set_category = set_category

        # create key part in DNN training: model, optimizer, dataloader
        self.model = model.to(device=self.device)
        self.optimizer = optimizer_parser(optimizer, self.model)
        self.dataloaders = dataloader_parser(dataloader, dataset, self.set_category)

        # register variables used in train/val/test process
        self.logger = logger
        self.train_epoch = train_epoch
        self.save_path = best_model_save_path

    def train(self, recorder, val_interval=2, val_start=0):
        model_saver = CC_Model_Saver(self.save_path)

        for epoch in range(1, 1 + self.train_epoch):
            self.model.train()
            recorder.reset_recorders()
            recorder.set_state('train', epoch)
            self.train_an_epoch(recorder)

            recorder.get_records()

            if epoch % val_interval == 0 and epoch >= val_start:
                self.model.eval()
                recorder.reset_recorders()
                recorder.set_state('test', epoch)
                self.val_an_epoch(recorder, 'test')
                mae, mse = recorder.get_records()

                model_saver.save(epoch, self.model, mae, mse)

            recorder.record(self.logger)
            recorder.reset_records()

    def train_an_epoch(self, recorder):
        raise NotImplementedError

    def val_an_epoch(self, recorder, set_name='val'):
        raise NotImplementedError


class Chf_trainer(Crowd_couting_trainer):
    def __init__(self, logger, model, optimizer: dict, dataset: str, dataloader: dict,
                 train_epoch: int = 1000, best_model_save_path: str = 'best_model.pth', sample_interval=8, im_size=512,
                 bandwidth=8, chf_step: int = 30, chf_tik: float = 0.01, is_dense: bool = False,
                 set_category=('train', 'val')):

        # register global parameters which is used in training/evaluation/test process
        self.chf_step = chf_step
        self.chf_tik = chf_tik

        # register parameters for dataloader
        dataloader['chf_step'] = chf_step
        dataloader['chf_tik'] = chf_tik
        dataloader['bandwidth'] = bandwidth

        # create loss
        self.lossfn = chfloss.Chfloss(chf_step, chf_tik, sample_interval, is_dense)
        '''
            if you want to use the noisy robust loss, use the following codes and comment out the above one-line code
        '''
        ' for General i.i.d. noise distribution, use the following codes, you can also set coeff as 1 rather than 0.5,' \
        '  see the comments at Central_Gaussian class '
        # likelihood = chf_likelihood.Central_Gaussian(chf_step, chf_tik, 'empirical_var.pt', 0.5)
        # self.lossfn = chfloss.Chf_Likelihood_Loss(chf_step, chf_tik, sample_interval, likelihood)
        ' for Gaussian noise distribution, use the following codes. You can also set coeff as 10 or 30 or else, see the' \
        'comments at Central_Gaussian_with_Gaussian_Noise class'
        # likelihood = chf_likelihood.Central_Gaussian_with_Gaussian_Noise(chf_step, chf_tik, 20, bandwidth)
        # self.lossfn = chfloss.Chf_Likelihood_Loss(chf_step, chf_tik, sample_interval, likelihood)

        # finish general setting in super class's initialization
        super(Chf_trainer, self).__init__(logger, model, optimizer, dataset, dataloader, train_epoch,
                                          best_model_save_path, sample_interval, im_size, set_category)

    def train_an_epoch(self, recorder):
        for inputs, chfs in self.dataloaders['train']:
            inputs = recorder.register('input', inputs.to(device=self.device))
            chfs = recorder.register('target', chfs.to(device=self.device))
            outputs = recorder.register('output', self.model(inputs))
            loss = recorder.register('loss', self.lossfn(outputs, chfs))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            recorder.register('people_count', chfs[:, self.chf_step, self.chf_step, 0])

            recorder.record_for_batch()

    def val_an_epoch(self, recorder, set_name='val'):
        for inputs, chfs, count, name in self.dataloaders[set_name]:
            inputs = recorder.register('input', inputs.to(device=self.device))
            recorder.register('people_count', count.to(device=self.device))
            with torch.set_grad_enabled(False):
                try:
                    outputs = self.model(inputs)
                except RuntimeError as e:
                    if 'CUDA' in str(e):
                        print(str(e))
                        print('solve it by moving to cpu.')
                        torch.cuda.empty_cache()
                        self.model.to(device='cpu')
                        outputs = self.model(inputs.cpu()).to(device=self.device)
                        self.model.to(device=self.device)
                    else:
                        raise RuntimeError(e)
                recorder.register('output', outputs)
                recorder.record_for_batch()

