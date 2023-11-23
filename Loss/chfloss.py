from torch import nn
import torch

class Chfloss(nn.Module):
    def __init__(self, chf_step: int, chf_tik: float, sample_step: float, is_dense: bool):
        '''

        Args:
            chf_step (single integer): how many steps the c.f. span in the plane (originate in the origin and expand to
            4 directions, up, down, left, right)

            chf_tik (single value): the span of each step, the final presented domain of the c.f. is decided by
                                    [-step×tik,step×tik)^2

            sample_step (single value): the sampling interval of the image plane. DNN outputs corresponds to the discrete
                                        sample in the image plane.
        '''
        super(Chfloss, self).__init__()
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.sample_step = sample_step
        self.is_dense = is_dense

        self.plane_shape = None
        self.real_template = None
        self.img_template = None

    def make_template(self, dnn_output: torch.Tensor):
        # construct spatial domain
        x_axis = torch.linspace(self.sample_step / 2, dnn_output.shape[-1] * self.sample_step - self.sample_step / 2,
                                dnn_output.shape[-1])
        y_axis = torch.linspace(self.sample_step / 2, dnn_output.shape[-2] * self.sample_step - self.sample_step / 2,
                                dnn_output.shape[-2])
        sample_coordinates = torch.stack([torch.repeat_interleave(x_axis, len(y_axis)), y_axis.repeat(len(x_axis))],
                                         dim=0).to(dtype=dnn_output.dtype, device=dnn_output.device)

        # constrcut ch.f. (frequential) domain
        plane = torch.cat(
            [torch.arange(-self.chf_step, self.chf_step).unsqueeze(0).expand(2 * self.chf_step,
                                                                             2 * self.chf_step).unsqueeze(
                2) * self.chf_tik,
             torch.arange(-self.chf_step, self.chf_step).unsqueeze(1).expand(2 * self.chf_step,
                                                                             2 * self.chf_step).unsqueeze(
                 2) * self.chf_tik],
            dim=2).to(dtype=dnn_output.dtype, device=dnn_output.device)

        # calculate the ch.f. template of the prediction
        angle = torch.matmul(plane, sample_coordinates)
        self.real_template = torch.cos(angle)
        self.img_template = torch.sin(angle)

    def forward(self, dnn_output: torch.Tensor, chf: torch.Tensor):
        if dnn_output.shape[-2:] != self.plane_shape:
            self.make_template(dnn_output)
            self.plane_shape = dnn_output.shape[-2:]

        # the ch.f. of the prediction is calculated by taking the sum of the DNN output value multiplying the corresponding template value
        flatten_output = dnn_output.transpose(-1, -2).contiguous().view(dnn_output.shape[0], -1)
        chf_real = (self.real_template * flatten_output[:, None, None, :]).sum(dim=3, keepdim=True)
        chf_img = (self.img_template * flatten_output[:, None, None, :]).sum(dim=3, keepdim=True)
        derived_chf = torch.cat([chf_real, chf_img], dim=3).to(dtype=chf.dtype, device=chf.device)


        '''On the `else' part, please note the exact L1norm loss should take the `dim=3', we use a improvement here to 
           prevent overfitting and decrease training variance, for the detailed reason, please pay attention to our 
           extended journal paper. For the non-super-dense dataset, we use the L2norm loss to further decrease training 
           variance. The decrease of both the L2norm and the modified L1norm will decrease the exact L1norm, while they 
           effectively prevent overfitting and increase training steadiness compared with L1norm loss.'''

        if not self.is_dense:
            loss = torch.sum(torch.norm((derived_chf - chf).view(chf.shape[0], -1), dim=1) * self.chf_tik)
        else:
            loss = torch.sum(torch.norm(derived_chf - chf, dim=2)) * self.chf_tik ** 2

        return loss / chf.shape[0]


class Chf_Likelihood_Loss(nn.Module):
    '''
        The loss used for noisy crowd counting
    '''
    def __init__(self, chf_step: int, chf_tik: float, sample_step: float, likelihood):
        '''

        Args:
            chf_step (single integer): how many steps the c.f. span in the plane (originate in the origin and expand to
            4 directions, up, down, left, right)

            chf_tik (single value): the span of each step, the final presented domain of the c.f. is decided by step×tik

            sample_step (single value): the sampling interval of the image plane. DNN outputs corresponds to the discrete
                                        sample in the image plane.
            bandwidth (single value): the bandwidth of the Gaussian used for approximating the density map. Note that this
                                      is the bandwidth of loss gaussian window multiplying the output.
        '''
        super(Chf_Likelihood_Loss, self).__init__()
        self.chf_step = chf_step
        self.chf_tik = chf_tik
        self.sample_step = sample_step

        self.likelihood = likelihood
        self.scale = 1

        self.plane_shape = None
        self.real_template = None
        self.img_template = None

    def make_template(self, dnn_output: torch.Tensor):
        # construct spatial domain
        x_axis = torch.linspace(self.sample_step / 2, dnn_output.shape[-1] * self.sample_step - self.sample_step / 2,
                                dnn_output.shape[-1])
        y_axis = torch.linspace(self.sample_step / 2, dnn_output.shape[-2] * self.sample_step - self.sample_step / 2,
                                dnn_output.shape[-2])
        sample_coordinates = torch.stack([torch.repeat_interleave(x_axis, len(y_axis)), y_axis.repeat(len(x_axis))],
                                         dim=0).to(dtype=dnn_output.dtype, device=dnn_output.device)

        # constrcut ch.f. (frequential) domain
        plane = torch.cat(
            [torch.arange(-self.chf_step, self.chf_step).unsqueeze(0).expand(2 * self.chf_step,
                                                                             2 * self.chf_step).unsqueeze(
                2) * self.chf_tik,
             torch.arange(-self.chf_step, self.chf_step).unsqueeze(1).expand(2 * self.chf_step,
                                                                             2 * self.chf_step).unsqueeze(
                 2) * self.chf_tik],
            dim=2).to(dtype=dnn_output.dtype, device=dnn_output.device)

        # calculate the ch.f. template of the prediction
        angle = torch.matmul(plane, sample_coordinates)
        self.real_template = torch.cos(angle)
        self.img_template = torch.sin(angle)


    def forward(self, dnn_output: torch.Tensor, chf: torch.Tensor):
        if dnn_output.shape != self.plane_shape:
            self.make_template(dnn_output)
            self.plane_shape = dnn_output.shape

        flatten_output = dnn_output.transpose(-1, -2).contiguous().view(dnn_output.shape[0], -1)
        chf_real = (self.real_template * flatten_output[:, None, None, :]).sum(dim=3, keepdim=True)
        chf_img = (self.img_template * flatten_output[:, None, None, :]).sum(dim=3, keepdim=True)


        derived_chf = torch.cat([chf_real, chf_img], dim=3).to(dtype=chf.dtype, device=chf.device)
        loss = self.likelihood.likelihood(derived_chf, chf, self.scale)
        return loss