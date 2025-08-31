import torch
import imageio.v2 as imageio
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        if opt.save == '.': opt.save = '../experiment/EXP/' + now
        self.dir = opt.save
        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)
        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')
        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(opt):
                f.write('{}: {}\n'.format(arg, getattr(opt, arg)))
            f.write('\n')

    def save(self, trainer, epochs, is_best=False, dual_model=False):
        epoch = trainer.get_last_epoch()
        trainer.model.save(self.dir, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        self.plot_psnr_ssim(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_ssim_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )
        if dual_model:
            dual_optimizers = {}
            for i in range(len(trainer.dual_optimizers)):
                dual_optimizers[i] = trainer.dual_optimizers[i]
            torch.save(
                dual_optimizers,
                os.path.join(self.dir, 'dual_optimizers.pt')
            )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr_ssim(self, epoch):
        # Skip plotting if no evaluation logs have been recorded
        if self.log.numel() == 0 or self.log.dim() < 2 or self.log.shape[1] < 2:
            self.write_log('No evaluation logs available; skipping PSNR/SSIM plot')
            return

        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.opt.data_test)
        fig = plt.figure(figsize=(10, 5))
        
        # PSNR plot
        plt.subplot(1, 2, 1)
        plt.title(label + ' - PSNR')
        for idx_scale, scale in enumerate([self.opt.scale[0]]):
            plt.plot(
                axis,
                self.log[:, idx_scale * 2].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        
        # SSIM plot
        plt.subplot(1, 2, 2)
        plt.title(label + ' - SSIM')
        for idx_scale, scale in enumerate([self.opt.scale[0]]):
            plt.plot(
                axis,
                self.log[:, idx_scale * 2 + 1].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('{}/test_{}_psnr_ssim.pdf'.format(self.dir, self.opt.data_test))
        plt.close(fig)
    
    def save_results_nopostfix(self, filename, sr, scale):
        apath = '{}/results/{}/x{}'.format(self.dir, self.opt.data_test, scale)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)
        
        normalized = sr[0].data.mul(255 / self.opt.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        
        # Convert numpy array to PIL Image
        if ndarr.shape[2] == 1:
            # Grayscale
            im = Image.fromarray(ndarr[:,:,0], mode='L')
        else:
            # RGB
            im = Image.fromarray(ndarr, mode='RGB')
        
        # Save the image
        im.save('{}.png'.format(filename))