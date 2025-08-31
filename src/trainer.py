import math
import time
import random
import numpy as np
import datetime
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
import copy
import contextlib
import os
from decimal import Decimal
from tqdm import tqdm
from src.data import Data
from src.helpers import analyze_window_sizes, analyze_window_sizes_gkd, process_images, process_gkd_images 
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def make_optimizer(opt, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {
        'betas': (opt.beta1, opt.beta2),
        'eps': opt.epsilon
    }
    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    
    return optimizer_function(trainable, **kwargs)


def make_dual_optimizer(opt, dual_models):
    dual_optimizers = []
    for dual_model in dual_models:
        temp_dual_optim = torch.optim.Adam(
            params=dual_model.parameters(),
            lr = opt.lr, 
            betas = (opt.beta1, opt.beta2),
            eps = opt.epsilon,
            weight_decay=opt.weight_decay)
        dual_optimizers.append(temp_dual_optim)
    
    return dual_optimizers


def make_scheduler(opt, my_optimizer):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        float(opt.epochs),
        eta_min=opt.eta_min
    )

    return scheduler


def make_dual_scheduler(opt, dual_optimizers):
    dual_scheduler = []
    for i in range(len(dual_optimizers)):
        scheduler = lrs.CosineAnnealingLR(
            dual_optimizers[i],
            float(opt.epochs),
            eta_min=opt.eta_min
        )
        dual_scheduler.append(scheduler)

    return dual_scheduler

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        print("the dimension of sr image is not equal to hr's! ")
        sr = sr[:,:,:hr.size(-2),:hr.size(-1)]
    diff = (sr - hr).data.div(rgb_range)

    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def calc_ssim(sr, hr, scale, rgb_range, benchmark=False):
    """Calculate SSIM (structural similarity) for the super-resolved and high-resolution images."""
    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        sr = sr[:, :, :hr.size(-2), :hr.size(-1)]
    
    sr = sr.div(rgb_range).clamp(0, 1)
    hr = hr.div(rgb_range).clamp(0, 1)

    shave = scale if benchmark else scale + 6

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
    return ssim_map.mean().item()

def print_gpu_memory_usage():
    # print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    # print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    # print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")
    
class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp, dual_model=False):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.dual_model = dual_model
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = make_optimizer(opt, self.model)
        self.scheduler = make_scheduler(opt, self.optimizer)
        if self.dual_model:
            self.dual_models = self.model.dual_models
            self.dual_optimizers = make_dual_optimizer(opt, self.dual_models)
            self.dual_scheduler = make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8
        # Enable AMP only on CUDA; disable on CPU/MPS for stability
        self.use_amp = (not self.opt.cpu) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Early Stopping Variables
        self.patience = opt.patience
        self.min_delta = opt.min_delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0


    def get_last_epoch(self):
        return self.scheduler.last_epoch

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = timer(), timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            if self.dual_model:
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].zero_grad()

            # forward
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else contextlib.nullcontext()
            with amp_ctx:
                sr = self.model(lr[0])
                
                if self.dual_model:
                    sr2lr = []
                    for i in range(len(self.dual_models)):
                        sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                        sr2lr.append(sr2lr_i)

                    # compute primary loss
                    loss_primary = self.loss(sr[-1], hr)
                    for i in range(1, len(sr)):
                        loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

                    # compute dual loss
                    loss_dual = self.loss(sr2lr[0], lr[0])
                    for i in range(1, len(self.scale)):
                        loss_dual += self.loss(sr2lr[i], lr[i])

                    # compute total loss
                    loss = loss_primary + self.opt.dual_weight * loss_dual
                else:
                    # compute primary loss
                    loss = self.loss(sr, hr)
            
            # if loss.item() < self.opt.skip_threshold * self.error_last:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()
            #self.optimizer.step()
            if self.dual_model:
                for i in range(len(self.dual_optimizers)):
                    if self.use_amp:
                        self.scaler.step(self.dual_optimizers[i])
                    else:
                        self.dual_optimizers[i].step()
            if self.use_amp:
                self.scaler.update()

            # else:
            #     print('Skip this batch {}! (Loss: {})'.format(
            #         batch + 1, loss.item()
            #     ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()       

        self.loss.end_log(len(self.loader_train))
        current_loss = self.loss.log[-1, -1]
        self.error_last = current_loss
        self.step()

        # Early stopping check
        # if current_loss < self.best_loss - self.min_delta:
        #     self.best_loss = current_loss
        #     self.epochs_no_improve = 0
        # else:
        #     self.epochs_no_improve += 1

        # if self.epochs_no_improve >= self.patience:
        #     print("Early stopping triggered")
        #     return True

        # return False

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 2))
        self.model.eval()

        timer_test = timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                eval_ssim = 0
                # tqdm_test = tqdm(self.loader_test, ncols=80)
                # for _, (lr, hr, filename) in enumerate(tqdm_test):
                for _, (lr, hr, filename) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else contextlib.nullcontext()
                    with amp_ctx:
                        sr = self.model(lr[0])
                    #sr = self.model(lr[0])
                        
                    if isinstance(sr, list): sr = sr[-1]

                    sr = quantize(sr, self.opt.rgb_range)

                    if not no_eval:
                        eval_psnr += calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_ssim += calc_ssim(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                self.ckp.log[-1, si * 2 + 1] = eval_ssim / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})\tSSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                    self.opt.data_test, s,
                    self.ckp.log[-1, si * 2],
                    best[0][si * 2],
                    best[1][si * 2] + 1,
                    self.ckp.log[-1, si * 2 + 1],
                    best[0][si * 2 + 1],
                    best[1][si * 2 + 1] + 1
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

    def test_early_stop(self):
        # Early-stop evaluation removed; keep a harmless stub to avoid breakage
        self.ckp.write_log('Early-stop evaluation disabled; returning 0.0')
        return 0.0

        
    def step(self):
        self.scheduler.step()
        if self.dual_model:
            for i in range(len(self.dual_scheduler)):
                self.dual_scheduler[i].step()

    def prepare(self, *args):
        # Use model-selected device (CUDA, MPS, or CPU)
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            if (not self.opt.cpu) and torch.cuda.is_available():
                device = torch.device('cuda')
            elif (not self.opt.cpu) and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs