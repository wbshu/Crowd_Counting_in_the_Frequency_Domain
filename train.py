from Model import vgg_19
from trainer import Chf_trainer
from Utils.recording import Recorder, Logger
import sys

if __name__ == '__main__':
    if (sys.argv[1].lower().find('shtc') >= 0 or sys.argv[1].lower().find(
            'shanghai') >= 0) and sys.argv[1].lower().find('a') >= 0:
        dataloader = {'datahandler': 'chf_rcrop', 'batch_size': 16, 'shorter_length_min': 384}
        train_epoch = 1000
        val_start = 0
        img_crop_size = 128
        is_dense = True
        sets = ('train', 'test')
    elif (sys.argv[1].lower().find('shtc') >= 0 or sys.argv[1].lower().find(
            'shanghai') >= 0) and sys.argv[1].lower().find('b') >= 0:
        dataloader = {'datahandler': 'chf_rcrop', 'batch_size': 8}
        train_epoch = 1000
        val_start = 0
        img_crop_size = 512
        is_dense = False
        sets = ('train', 'test')
    elif sys.argv[1].lower().find('qnrf') >= 0:
        dataloader = {'datahandler': 'chf_rcrop', 'batch_size': 32, 'shorter_length_min': 384, 'shorter_length_max': 1536}
        train_epoch = 1000
        val_start = 0
        img_crop_size = 384
        is_dense = True
        sets = ('train', 'test')
    elif sys.argv[1].lower().startswith('jhu'):
        dataloader = {'datahandler': 'hard_dish_chf_rcrop', 'batch_size': 64}
        train_epoch = 800
        val_start = 400
        img_crop_size = 384
        is_dense = False
        sets = ('train', 'val')
    elif sys.argv[1].lower().find('nwpu') >= 0:
        dataloader = {'datahandler': 'hard_dish_chf_rcrop', 'batch_size': 64}
        train_epoch = 900
        val_start = 800
        img_crop_size = 384
        is_dense = False
        sets = ('train', 'val')
    else:
        raise NotImplementedError

    config = {
        'best_model_save_path': sys.argv[1] + '/' + sys.argv[2] + '.pth',
        "optimizer": {'optimizer': 'adam', 'lr': 1e-5, 'weight_decay': 1e-4},
        "dataset": sys.argv[1],
        "sample_interval": 8,
        "bandwidth": 8,
        "chf_step": 30,
        "chf_tik": 0.01
    }

    t = Chf_trainer(Logger(), vgg_19.vgg19(), config['optimizer'],
                    config["dataset"], dataloader, train_epoch,
                    best_model_save_path=config['best_model_save_path'],
                    sample_interval=config["sample_interval"], im_size=img_crop_size,
                    bandwidth=config["bandwidth"],
                    chf_step=config["chf_step"], chf_tik=config["chf_tik"], is_dense=is_dense,
                    set_category=sets
                    )
    recorder = Recorder()
    recorder.basic_setting()
    t.train(recorder, val_start=val_start)
