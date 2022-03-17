import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader

import Dataset.dataloader as dl
from Dataset.preprocessor import Directory_path
from Utils import metric_measure as mm


class Chf_Model_Test():
    def __init__(self, model, parameter_path):
        '''

        :param model: network model
        :param parameter_path:  model parameter's saving path

        '''
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(parameter_path))
            self.model = model.to(device='cuda')
        else:
            self.model = model.load_state_dict(torch.load(parameter_path, map_location=torch.device('cpu')))

        self.model.eval()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gray_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def evaluate_on_test_set(self, dataset, ofile='model_test_result.txt', set='test', min_size=0, max_size=2048,
                             is_gray=False):
        with torch.no_grad():
            path_prefix, img_path_suffix, dot_path_suffix = Directory_path.prefix_suffix(dataset)
            epoch_mae = mm.Average_Metric()
            epoch_mse = mm.Average_Metric()
            data = DataLoader(dl.ChfData_RCrop(path_prefix + set + img_path_suffix,
                                               path_prefix + set + dot_path_suffix, set, min_size=min_size,
                                               max_size=max_size, device=self.device, is_gray=is_gray), num_workers=8,
                              pin_memory=True)

            for inputs, chfs, count, name in data:
                inputs = inputs.to(device=self.device)
                try:
                    outputs = self.model(inputs)
                except RuntimeError as e:
                    if 'CUDA' in str(e):
                        print(e)
                        print(name)
                        torch.cuda.empty_cache()
                        img_tensor = inputs.to(device='cpu')
                        self.model.to(device='cpu')
                        outputs = self.model(img_tensor)
                        self.model.to(device='cuda')
                        outputs = outputs.to(device='cuda')

                predict = torch.sum(outputs)
                res = predict - count[0].item()
                res = res.cpu().numpy()
                epoch_mse.update(res * res)
                epoch_mae.update(abs(res))
                with open(ofile, "at") as f:
                    print(name[0], count[0].item(), predict.cpu().numpy().item(), file=f)

            print(epoch_mae.get_avg(), np.sqrt(epoch_mse.get_avg()))

            return epoch_mae.get_avg(), np.sqrt(epoch_mse.get_avg())

    def evaluate_on_NWPU_test(self, min_size=0, max_size=2048, set='test', file='result_nwpu_test.txt'):
        with torch.no_grad():
            path_prefix, img_path_suffix, dot_path_suffix = Directory_path.prefix_suffix('nwpu')
            data = DataLoader(dl.NWPU_Test_Loader(path_prefix + set + img_path_suffix, min_size, max_size),
                              num_workers=8, pin_memory=True)
            with open(file, "at") as f:
                for inputs, name in data:
                    inputs = inputs.to(device=self.device)
                    try:
                        outputs = self.model(inputs)
                    except RuntimeError as e:
                        if 'CUDA' in str(e):
                            print(e)
                            print(name)
                            torch.cuda.empty_cache()
                            img_tensor = inputs.to(device='cpu')
                            self.model.to(device='cpu')
                            outputs = self.model(img_tensor)
                            self.model.to(device='cuda')
                            outputs = outputs.to(device='cuda')

                    count = torch.sum(outputs).cpu().numpy().item()
                    print(name[0], count, file=f)
