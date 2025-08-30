import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F


def calc_ssim(sr, hr, batch_size, scale=4, rgb_range=255):
    """Calculate SSIM (structural similarity) for the super-resolved and high-resolution images."""
    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        sr = sr[:, :, :hr.size(-2), :hr.size(-1)]

    sr = sr.div(rgb_range).clamp(0, 1)
    hr = hr.div(rgb_range).clamp(0, 1)

    shave = scale + 6

    if sr.size(-1) > 2 * shave:
        sr = sr[..., shave:-shave, shave:-shave]
        hr = hr[..., shave:-shave, shave:-shave]
    else:
        sr = sr[..., 1:-1, 1:-1]
        hr = hr[..., 1:-1, 1:-1]

    if sr.size(1) > 1:
        convert = torch.tensor([[65.738, 129.057, 25.064]], dtype=sr.dtype, device=sr.device).view(1, 3, 1, 1) / 256
        sr = (sr * convert).sum(dim=1, keepdim=True)
        hr = (hr * convert).sum(dim=1, keepdim=True)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = torch.ones(1, 1, 11, 11, dtype=sr.dtype, device=sr.device) / 121

    sr = sr.to(hr.dtype)  # Ensure sr and hr are the same type
    kernel = kernel.to(hr.dtype)  # Ensure kernel is the same type as hr

    mu1 = F.conv2d(sr, kernel, padding=5)
    mu2 = F.conv2d(hr, kernel, padding=5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(sr ** 2, kernel, padding=5) - mu1_sq
    sigma2_sq = F.conv2d(hr ** 2, kernel, padding=5) - mu2_sq
    sigma12 = F.conv2d(sr * hr, kernel, padding=5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return (1 - ssim_map).sum()/batch_size

class SSIMLoss(nn.Module):
    def __init__(self, args):
        super(SSIMLoss, self).__init__()
        self.data_range = args.rgb_range
        self.batch_size = args.batch_size

    def forward(self, sr, hr):
        return calc_ssim(sr, hr, self.batch_size, 4, rgb_range=self.data_range)

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, sr, hr):
        mse = nn.MSELoss(reduction='mean')(sr, hr)
        psnr = 10 * torch.log10((255 ** 2)/(mse+1e-8))
        return -psnr.mean() 
        
class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')

            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss(reduction='mean')
            elif loss_type == 'PSNR':
                loss_function = PSNRLoss()
            elif loss_type == 'SSIM':
                loss_function = SSIMLoss(args)
            else:
                assert False, f"Unsupported loss type: {loss_type:s}"
            
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def save(self, apath):
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))