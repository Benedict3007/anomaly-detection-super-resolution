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
import os
from decimal import Decimal
from tqdm import tqdm
from data import Data
from helpers import analyze_window_sizes, analyze_window_sizes_gkd, process_images, process_gkd_images 
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
        self.scaler = torch.cuda.amp.GradScaler()

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
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
            self.scaler.scale(loss).backward()
            #loss.backward()                
            self.scaler.step(self.optimizer)
            #self.optimizer.step()
            if self.dual_model:
                for i in range(len(self.dual_optimizers)):
                    self.scaler.step(self.dual_optimizers[i])
                    #self.dual_optimizers[i].step()
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

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 2))
        self.model.eval()
        scaling = max(self.scale)
        scale = scaling
        # Parameter
        if self.opt.slurm:
            workspace = '/europa/hpc-homes/bd6102s/workspace'
        else:
            workspace = 'workspace'
        no_augment = self.opt.no_augment
        epochs = 1000
        batch_size = 1 
        patch_size = self.opt.patch_size
        resolution = patch_size
        img_size = patch_size // scaling
        n_colors = self.opt.n_colors
        dataset = self.opt.dataset
        classe = self.opt.classe

        if dataset == 'mvtec':
            data_dir_good = f'{workspace}/mvtec_anomaly_detection_modified/{classe}/test/HR_{resolution}/good'
            data_dir_bad = f'{workspace}/mvtec_anomaly_detection_modified/{classe}/test/HR_{resolution}/bad'
            dataset_length = 256
        elif dataset == 'gkd':
            data_dir_good = f'{workspace}/gkd/{classe}/test/HR_{resolution}/good'
            data_dir_bad = f'{workspace}/gkd/{classe}/test/HR_{resolution}/bad'
            dataset_length = 2048
        else:
            print("Not the right dataset!")

        now = datetime.datetime.now()
        date_string = now.strftime("%H:%M:%S")

        data_range: str='' # train test data range
        test_every = dataset_length // batch_size
        print_every = test_every
        n_threads = 4
        model_type = self.opt.model_name
        model_name = f"{model_type}_{classe}_{patch_size}_{scaling}_earlyStopping_{date_string}"

        if model_type == 'drn-l':
            save = f'{workspace}/images/drn-l/{model_name}/'
            opt_good = copy.deepcopy(self.opt)
            # opt_good.scale = scale
            opt_good.save_results = False
            opt_good.test_only = True
            opt_good.no_augment = True
            opt_good.batch_size = batch_size
            opt_good.data_dir = data_dir_good
            opt_good.save = save
            opt_good.data_range = data_range
            opt_good.test_every = test_every 
            opt_good.print_every = print_every 

            opt_bad = copy.deepcopy(self.opt)
            # opt_bad.scale = scale 
            opt_bad.save_results = False
            opt_bad.test_only = True
            opt_bad.no_augment = True
            opt_bad.batch_size = batch_size
            opt_bad.data_dir = data_dir_bad
            opt_bad.save = save
            opt_bad.data_range = data_range
            opt_bad.test_every = test_every 
            opt_bad.print_every = print_every 
            # opt_good = setup_opt_drn(opt_good, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir_good, save, data_range, test_every, print_every, patience, n_threads, model_name)
            # opt_bad = setup_opt_drn(opt_bad, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir_bad, save, data_range, test_every, print_every, patience, n_threads, model_name)
        elif model_type == 'drct':
            save = f'{workspace}/images/drct/{model_name}/'
            opt_good = copy.deepcopy(self.opt)
            # opt_good.scale = scale
            opt_good.save_results = False
            opt_good.test_only = True
            opt_good.no_augment = True
            opt_good.batch_size = batch_size
            opt_good.data_dir = data_dir_good
            opt_good.save = save
            opt_good.data_range = data_range
            opt_good.test_every = test_every 
            opt_good.print_every = print_every 

            opt_bad = copy.deepcopy(self.opt)
            # opt_bad.scale = scale
            opt_bad.save_results = False
            opt_bad.test_only = True
            opt_bad.no_augment = True
            opt_bad.batch_size = batch_size
            opt_bad.data_dir = data_dir_bad
            opt_bad.save = save
            opt_bad.data_range = data_range
            opt_bad.test_every = test_every 
            opt_bad.print_every = print_every 
            # opt_good = setup_opt_drn(opt_good, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir_good, save, data_range, test_every, print_every, patience, n_threads, model_name)
            # opt_bad = setup_opt_drn(opt_bad, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir_bad, save, data_range, test_every, print_every, patience, n_threads, model_name)
        else:
            print("Model_Type unknown!")
            
        if model_type == 'drn-l':
            loader_good = Data(opt_good)
            loader_bad = Data(opt_bad)
            
            test_loader_good = loader_good.loader_test
            test_loader_bad = loader_bad.loader_test
            filepath_good = f"{workspace}/images/drn-l/{model_name}/predicted_images/good"
            filepath_bad = f"{workspace}/images/drn-l/{model_name}/predicted_images/bad"
            os.makedirs(filepath_good, exist_ok=True)
            os.makedirs(filepath_bad, exist_ok=True)
        elif model_type == 'drct':
            loader_good = Data(opt_good)
            loader_bad = Data(opt_bad)
            
            test_loader_good = loader_good.loader_test
            test_loader_bad = loader_bad.loader_test
            filepath_good = f"{workspace}/images/drct/{model_name}/predicted_images/good"
            filepath_bad = f"{workspace}/images/drct/{model_name}/predicted_images/bad"
            os.makedirs(filepath_good, exist_ok=True)
            os.makedirs(filepath_bad, exist_ok=True)
        else:
            print("Model_Type unknown!")


        with torch.no_grad():
            scale = opt_good.scale
            for si, s in enumerate([scale]):
                eval_psnr = 0
                # tqdm_test = tqdm(test_loader_good, ncols=80)
                for _, (lr, hr, filename) in enumerate(test_loader_good):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr[0])
                    
                    if isinstance(sr, list): sr = sr[-1]
                    
                    sr = quantize(sr, opt_good.rgb_range)
                                                    
                    # Save super-resolved image
                    filename = os.path.join(filepath_good, filename)
                    normalized = sr[0].data.mul(255 / opt_good.rgb_range)
                
                    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                
                    # imageio.imwrite('{}.png'.format(filename), ndarr)
                    # Convert to PIL Image and save
                    if ndarr.ndim == 2:
                        img = Image.fromarray(ndarr, mode='L')
                    elif ndarr.ndim == 3:
                        if ndarr.shape[2] == 1:
                            img = Image.fromarray(ndarr.squeeze(), mode='L')
                        elif ndarr.shape[2] == 3:
                            img = Image.fromarray(ndarr, mode='RGB')
                        elif ndarr.shape[2] == 4:
                            img = Image.fromarray(ndarr, mode='RGBA')
                        else:
                            raise ValueError(f"Unexpected number of channels: {ndarr.shape[2]}")
                    else:
                        raise ValueError(f"Unexpected number of dimensions: {ndarr.ndim}")
                    
                    img.save(f'{filename}.png')

        with torch.no_grad():
            scale = opt_bad.scale
            for si, s in enumerate([scale]):
                eval_psnr = 0
                # tqdm_test = tqdm(test_loader_bad, ncols=80)
                for _, (lr, hr, filename) in enumerate(test_loader_bad):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    # Super-resolution model prediction
                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]
                    sr = quantize(sr, opt_bad.rgb_range)
                    
                    # Save super-resolved image
                    filename = os.path.join(filepath_bad, filename)
                    normalized = sr[0].data.mul(255 / opt_bad.rgb_range)
                    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                    # imageio.imwrite('{}.png'.format(filename), ndarr)
                    if ndarr.ndim == 2:
                        img = Image.fromarray(ndarr, mode='L')
                    elif ndarr.ndim == 3:
                        if ndarr.shape[2] == 1:
                            img = Image.fromarray(ndarr.squeeze(), mode='L')
                        elif ndarr.shape[2] == 3:
                            img = Image.fromarray(ndarr, mode='RGB')
                        elif ndarr.shape[2] == 4:
                            img = Image.fromarray(ndarr, mode='RGBA')
                        else:
                            raise ValueError(f"Unexpected number of channels: {ndarr.shape[2]}")
                    else:
                        raise ValueError(f"Unexpected number of dimensions: {ndarr.ndim}")
                    
                    img.save(f'{filename}.png')


        if dataset == 'mvtec':
            if model_type == 'drn-l':
                good_original_folder = f"{workspace}/mvtec_anomaly_detection_modified/{classe}/test/HR_{resolution}/good/HR"
                bad_original_folder = f"{workspace}/mvtec_anomaly_detection_modified/{classe}/test/HR_{resolution}/bad/HR"
                good_reconstructed_folder = f"{workspace}/images/drn-l/{model_name}/predicted_images/good"
                bad_reconstructed_folder = f"{workspace}/images/drn-l/{model_name}/predicted_images/bad"
                log_file_path = f"{workspace}/images/drn-l/{model_name}/scores.txt"
            elif model_type == 'drct':
                good_original_folder = f"{workspace}/mvtec_anomaly_detection_modified/{classe}/test/HR_{resolution}/good/HR"
                bad_original_folder = f"{workspace}/mvtec_anomaly_detection_modified/{classe}/test/HR_{resolution}/bad/HR"
                good_reconstructed_folder = f"{workspace}/images/drct/{model_name}/predicted_images/good"
                bad_reconstructed_folder = f"{workspace}/images/drct/{model_name}/predicted_images/bad"
                log_file_path = f"{workspace}/images/drct/{model_name}/scores.txt"
            else:
                print("Unknown Model Type")
        elif dataset == 'gkd':
            if model_type == 'drn-l':
                good_original_folder = f"{workspace}/gkd/{classe}/test/HR_{resolution}/good/HR"
                bad_original_folder = f"{workspace}/gkd/{classe}/test/HR_{resolution}/bad/HR"
                good_reconstructed_folder = f"{workspace}/images/drn-l/{model_name}/predicted_images/good"
                bad_reconstructed_folder = f"{workspace}/images/drn-l/{model_name}/predicted_images/bad"
                log_file_path = f"{workspace}/images/drn-l/{model_name}/scores.txt"
            elif model_type == 'drct':
                good_original_folder = f"{workspace}/gkd/{classe}/test/HR_{resolution}/good/HR"
                bad_original_folder = f"{workspace}/gkd/{classe}/test/HR_{resolution}/bad/HR"
                good_reconstructed_folder = f"{workspace}/images/drct/{model_name}/predicted_images/good"
                bad_reconstructed_folder = f"{workspace}/images/drct/{model_name}/predicted_images/bad"
                log_file_path = f"{workspace}/images/drct/{model_name}/scores.txt"
            else:
                print("Unknown Model Type")
        else: 
            print("Unknown Dataset!")        

        if dataset == 'mvtec':
            chosen_window_size = self.opt.ssim_window_size

            y_true, y_scores_ssim, y_scores_mse, y_scores_psnr = process_images(
                good_original_folder, good_reconstructed_folder,
                bad_original_folder, bad_reconstructed_folder,
                log_file_path, chosen_window_size
            )

            r_a_ssim = roc_auc_score(y_true, y_scores_ssim)
            r_a_mse = roc_auc_score(y_true, y_scores_mse)
            r_a_psnr = roc_auc_score(y_true, y_scores_psnr)
        elif dataset == 'gkd':
            chosen_window_size = self.opt.ssim_window_size

            y_true, y_scores_ssim, y_scores_mse, y_scores_psnr = process_gkd_images(
                good_original_folder, good_reconstructed_folder,
                bad_original_folder, bad_reconstructed_folder,
                log_file_path, chosen_window_size
            )
            r_a_ssim = roc_auc_score(y_true, y_scores_ssim)
            r_a_mse = roc_auc_score(y_true, y_scores_mse)
            r_a_psnr = roc_auc_score(y_true, y_scores_psnr)
        else: 
            print("Unknown Dataset!")    
        
        best_auc = max(r_a_ssim, r_a_mse, r_a_psnr)
        print(f"Best AUC Score: {best_auc:.4f}")
        return best_auc

        
    def step(self):
        self.scheduler.step()
        if self.dual_model:
            for i in range(len(self.dual_scheduler)):
                self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

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