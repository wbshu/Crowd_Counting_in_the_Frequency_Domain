import torch


class Chf_Likelihood():
    def likelihood(self, input, target, scale):
        raise NotImplementedError


class Central_Gaussian(Chf_Likelihood):
    '''
        General i.i.d. noise-robust window for noisy crowd counting. Please refer to my journal paper sec.III.E
    '''
    def __init__(self, chf_step, chf_tik, var: str, coeff):
        '''

        :param chf_step:
        :param chf_tik:
        :param var: the empirical error variance map file
        :param coeff: usually we set it as 1 or 0.5, it depends on the noise level, if the annotation noise is large,
                      then set it as 1, otherwise set it as 0.5
        '''
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.h_r = torch.load(var).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.coeff = coeff

    def likelihood(self, input, target, scale):
        people_count = target[:, int(target.shape[1] / 2), int(target.shape[2] / 2), 0]
        if type(scale) is not int:
            scale = scale.reshape(self.h_r.shape[0], self.h_r.shape[1])
        var = (scale ** 2 * self.h_r).unsqueeze(0) * people_count.unsqueeze(-1).unsqueeze(-1)
        loss = torch.norm((input - target), dim=-1) / torch.sqrt(self.coeff*var + 1)
        loss = torch.sum(loss) / target.shape[0]

        return loss

class Central_Gaussian_with_Gaussian_Noise(Chf_Likelihood):
    '''
        i.i.d. Gaussian noise-robust window for noisy crowd counting. Please refer to my journal paper sec.III.E
    '''
    def __init__(self, chf_step, chf_tik, noise_bandwidth, ground_truth_bandwidth):
        '''
        :param chf_step:
        :param chf_tik:
        :param noise_bandwidth: the bandwidth of the Gaussian distribution of the annotation noise, the annotation noise
                                is bigger, the parameter should be bigger, usually we set it as 10, 20 or 30.
        :param ground_truth_bandwidth: the bandwidth of the ground truth density map
        '''
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        plane = torch.cat(
            [torch.arange(-self.chf_step, self.chf_step).unsqueeze(0).expand(2 * self.chf_step,
                                                                             2 * self.chf_step).unsqueeze(
                2) * self.chf_tik,
             torch.arange(-self.chf_step, self.chf_step).unsqueeze(1).expand(2 * self.chf_step,
                                                                             2 * self.chf_step).unsqueeze(
                 2) * self.chf_tik],
            dim=2).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        original_chf_factor = torch.exp(-1 / 2 * (plane * plane).sum(dim=2, keepdim=False) * ground_truth_bandwidth ** 2)
        noise_chf_factor = torch.exp(-1 / 2 * (plane * plane).sum(dim=2, keepdim=False) * noise_bandwidth ** 2)
        self.h_r = (1-noise_chf_factor)*original_chf_factor

    def likelihood(self, input, target, scale):
        people_count = target[:, int(target.shape[1] / 2), int(target.shape[2] / 2), 0]
        if type(scale) is not int:
            scale = scale.reshape(self.h_r.shape[0], self.h_r.shape[1])
        var = (scale ** 2 * self.h_r).unsqueeze(0) * people_count.unsqueeze(-1).unsqueeze(-1)
        var = var.to(dtype=input.dtype, device=input.device)
        loss = torch.norm((input - target), dim=-1) / torch.sqrt(var + 1)
        loss = torch.sum(loss) / target.shape[0]

        return loss