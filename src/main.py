import torch
import matplotlib
import random
import numpy as np
import datetime
import time
from dataclasses import dataclass

from src.checkpoint import Checkpoint
from src.model import Model
from src.data import Data
from src.loss import Loss
from src.trainer import Trainer

matplotlib.use('Agg')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)

@dataclass
class DRN:
    model_name: str='drn-l'
    n_threads: int=-2 # number of threads for data loading
    cpu: bool=False # use cpu only
    n_GPUs: int=1 # number of GPUs
    seed: int=1 # random seed
    data_dir: str='./workspace/gkd/DC2/unlabeled/HR_512_grayscale/' # dataset directory
    data_train: str='' # train dataset name
    data_test: str='' # test dataset name
    data_range: str='1-224/225-280' # train test data range
    scale: int=4 # super resolution scale
    patch_size: int=512 # output patch size
    rgb_range: int=255 # maximum value of RGB
    n_colors: int=1 # number of color channels to use
    no_augment: bool=False # do not use data augmentation
    pre_train: str='.' # pre-trained model directory
    pre_train_dual: str='.' # pre-trained dual model directory
    n_blocks: int=40 # number of residual blocks, 16|30|40|80
    n_feats: int=20 # number of feature maps 
    negval: float=0.2 # Negative value parameter for Leaky ReLU
    test_every: int=10 # do test per every N batches
    epochs: int=10 # number of epochs to train
    batch_size: int=4 # input batch_size for training
    self_ensemble: bool=False # use self-ensemble method for test
    test_only: bool=False # set this option to test the model
    lr: float=1e-4 # learning rate
    eta_min: float=1e-7 # eta_min learning rate
    beta1: float=0.9 # ADAM beta1
    beta2: float=0.999 # ADAM beta2
    epsilon: float=1e-8 # ADAM epsilon for numerical stability
    weight_decay: float=1e-8 # weight decay
    loss: str='1*L1' # loss function configuration, L1/MSE
    # skip_threshold: float=1e6 # skipping batch that has large error
    skip_threshold: float=1.5 # skipping batch that has large error
    dual_weight: float=0.1 # the weight of dual loss
    save: str='./workspace/experiment/drn-l/gkd_dc2_unlabeld_X4_10_grayscale/' # file name to save
    print_every: int=10 # how many batches to wait before logging training status
    save_results: bool=True# save output results
    dual: bool=True
    patience: int=10
    dataset: str=''
    classe: str=''
    slurm: bool=False
    ssim_window_size: int=11
    best_auc: float=1.0

@dataclass
class DRCT:
    model_name: str='drct'
    n_threads: int=1 # number of threads for data loading
    cpu: bool=False # use cpu only
    n_GPUs: int=1 # number of GPUs
    seed: int=1 # random seed
    data_dir: str='./workspace/gkd/DC2/unlabeled/HR_512_grayscale/' # dataset directory
    data_train: str='' # train dataset name
    data_test: str='' # test dataset name
    data_range: str='1-260/261-299' # train test data range
    scale: int=4 # super resolution scale
    patch_size: int=512 # output patch size
    rgb_range: int=255 # maximum value of RGB
    n_colors: int=1 # number of color channels to use
    no_augment: bool=False# do not use data augmentation
    pre_train: str='.' # pre-trained model directory
    pre_train_dual: str='.' # pre-trained dual model directory
    negval: float=0.2 # Negative value parameter for Leaky ReLU
    test_every: int=30 # do test per every N batches
    epochs: int=10 # number of epochs to train
    batch_size: int=2 # input batch_size for training
    self_ensemble: bool=False # use self-ensemble method for test
    test_only: bool=False # set this option to test the model
    lr: float=1e-4 # learning rate
    eta_min: float=1e-7 # eta_min learning rate
    beta1: float=0.9 # ADAM beta1
    beta2: float=0.999 # ADAM beta2
    epsilon: float=1e-8 # ADAM epsilon for numerical stability
    loss: str='1*L1' # loss function configuration, L1/MSE
    skip_threshold: float=1e6 # skipping batch that has large error
    dual_weight: float=0.1 # the weight of dual loss
    save: str='./workspace/experiment/drct/gkd_dc2_unlabeled_X4_10_test_grayscale/' # file name to save
    print_every: int=10 # how many batches to wait before logging training status
    save_results: bool=True# save output results
    dual: bool=False
    upscale: int=4
    img_size: int=128
    window_size: int=16
    compress_ratio: int=3
    squeeze_factor: int=30
    conv_scale: float=0.01
    overlap_ratio: float=0.5
    img_range: float=1.0
    depths: list=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    embed_dim: int=180
    num_heads: list=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    mlp_ratio: int=2
    upsampler: str='pixelshuffle'
    resi_connection: str='1conv'
    ema_decay: float=0.999
    lr: float=2e-4
    weight_decay: float=0.0
    betas: list=(0.9, 0.99)
    patience: int=10
    dataset: str=''
    classe: str=''
    slurm: bool=False
    ssim_window_size: int=11
    best_auc: float=1.0

def setup_opt_drn(opt, best_auc, ssim_window_size, dataset, classe, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, pre_trained, pre_trained_dual, loss):
    opt.scale = scale
    opt.scale = [pow(2, s+1) for s in range(int(np.log2(opt.scale)))]

    if scale == 2:
        opt.n_blocks = 44
        opt.n_feats = 40
    elif scale == 4:
        opt.n_blocks = 40
        opt.n_feats = 20
    elif scale == 8:
        opt.n_blocks = 36
        opt.n_feats = 10
    else:
        print(f"No setup for this scale: {scale}")

    opt.no_augment = no_augment
    opt.n_colors = n_colors
    opt.epochs = epochs
    opt.batch_size = batch_size
    opt.patch_size = patch_size
    opt.data_dir = data_dir
    opt.save = save
    opt.test_every = test_every
    opt.print_every = print_every
    opt.patience = patience
    opt.min_delta = min_delta
    opt.n_threads = n_threads
    opt.pre_train = pre_trained
    opt.pre_train_dual = pre_trained_dual
    opt.loss = loss
    opt.dataset = dataset
    opt.classe = classe
    opt.slurm = slurm
    opt.ssim_window_size = ssim_window_size
    opt.best_auc = best_auc

    return opt

def setup_opt_drct(opt, best_auc, ssim_window_size, dataset, classe, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, img_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, pre_trained, loss):
    opt.scale = scale
    opt.upscale = scale
    opt.scale = [opt.scale] 
    opt.no_augment = no_augment
    opt.n_colors = n_colors
    opt.epochs = epochs
    opt.batch_size = batch_size
    opt.patch_size = patch_size
    opt.data_dir = data_dir
    opt.data_range = data_range
    opt.save = save
    opt.test_every = test_every
    opt.print_every = print_every
    opt.img_size = img_size
    opt.patience = patience
    opt.min_delta = min_delta
    opt.n_threads = n_threads
    opt.pre_train = pre_trained
    opt.window_size = img_size // 4
    opt.loss = loss
    opt.dataset = dataset
    opt.classe = classe
    opt.slurm = slurm
    opt.ssim_window_size = ssim_window_size
    opt.best_auc = best_auc
    
    return opt
    
def train_drn(opt_drn):
    set_seed(opt_drn.seed)
    
    checkpoint_drn = Checkpoint(opt_drn)

    if checkpoint_drn.ok:
        loader = Data(opt_drn)
        model = Model(opt_drn, checkpoint_drn, dual_model=True)
        loss = Loss(opt_drn, checkpoint_drn) if not opt_drn.test_only else None
        t = Trainer(opt_drn, loader, model, loss, checkpoint_drn, dual_model=True)
        start_time = time.time()

        # Train for the configured number of epochs; no early stopping
        while not t.terminate():
            t.train()
    

        # while not t.terminate():
        #     early_stop = t.train()
        #     if early_stop:
        #         print("Early stopping triggered. Ending training.")
        #         break
        #     t.test()

        # while not t.terminate():
        #     t.train()
        #     t.test()

        print("Training completed")
        end_time = time.time()
        checkpoint_drn.write_log(f"Total Training Time: {((end_time -  start_time)/3600):.2f}")
        checkpoint_drn.save(t, opt_drn.epochs, is_best=True, dual_model=True)
        checkpoint_drn.done()

def train_drct(opt_drct):
    set_seed(opt_drct.seed)
    
    checkpoint_drct = Checkpoint(opt_drct)

    if checkpoint_drct.ok:
        loader = Data(opt_drct)
        model = Model(opt_drct, checkpoint_drct, dual_model=False)
        loss = Loss(opt_drct, checkpoint_drct) if not opt_drct.test_only else None
        t = Trainer(opt_drct, loader, model, loss, checkpoint_drct, dual_model=False)
        start_time = time.time()

        # Train for the configured number of epochs; no early stopping
        while not t.terminate():
            t.train()

        # while not t.terminate():
        #     early_stop = t.train()
        #     if early_stop:
        #         print("Early stopping triggered. Ending training.")
        #         break
        #     t.test()

        # while not t.terminate():
        #     t.train()
        #     t.test()

        print("Training completed")
        end_time = time.time()
        checkpoint_drct.write_log(f"Total Training Time: {((end_time -  start_time)/3600):.2f}")
        checkpoint_drct.save(t, opt_drct.epochs, is_best=True, dual_model=False)
        checkpoint_drct.done()

if __name__ == "__main__":
    slurm = False
    # Defaults for local runs (used by evaluation utilities)
    best_auc = 0.0
    ssim_window_size = 11
    
    model_type = 'drct'
    pre_train = False
    
    mvtec = ['grid']
    gkd = ['DC2']
    datasets = ['mvtec']
    scaling = [4]
    resolutions = [128]
    for reso in resolutions:
        for ds in datasets:
            if ds == 'mvtec':
                classes = mvtec
            elif ds == 'gkd':
                classes = gkd
            elif ds == 'gkd_large':
                classes = gkd
            else:
                raise ValueError(f"Unknown dataset: {ds}")

            for class_name in classes:
                torch.cuda.empty_cache()
                # Parameter
                img_resolution = reso
                scale = scaling[0]
                no_augment = False
                print(f"Model: {model_type}")
                print(f"Dataset: {ds}")
                print(f"Class: {class_name}")
                print(f"Resolution: {reso}")
                print(f"Scale: {scale}")
                
                if class_name == "carpet":
                    n_colors = 3
                    if model_type == 'drn-l' and scale == 2:
                        pre_train = False
                else:
                    n_colors = 1
                    if model_type == 'drn-l':
                        pre_train = False

                if ds == 'mvtec':
                    # small local run
                    epochs = 2
                    batch_size = 4
                elif ds == 'gkd':
                    epochs = 500
                    batch_size = 64
                elif ds == 'gkd_large':
                    epochs = 500
                    batch_size = 64
                else:
                    raise ValueError(f"Unknown dataset: {ds}")

                patch_size = img_resolution
                img_size = img_resolution // scale

                now = datetime.datetime.now()

                if slurm:
                    date_string = now.strftime("%H:%M:%S")
                    if ds == 'mvtec':
                        data_dir = f'/europa/hpc-homes/bd6102s/workspace/mvtec_anomaly_detection_modified/{class_name}/train/HR_{img_resolution}'
                        if model_type == 'drn-l':
                            save = f'/europa/hpc-homes/bd6102s/workspace/experiment/drn-l/mvtec_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        elif model_type == 'drct':
                            save = f'/europa/hpc-homes/bd6102s/workspace/experiment/drct/mvtec_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        else:
                            raise ValueError(f"Unknown Model Type: {model_type}")
                    elif ds == 'gkd':
                        data_dir = f'/europa/hpc-homes/bd6102s/workspace/gkd/{class_name}/train/HR_{img_resolution}'
                        if model_type == 'drn-l':
                            save = f'/europa/hpc-homes/bd6102s/workspace/experiment/drn-l/gkd_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        elif model_type == 'drct':
                            save = f'/europa/hpc-homes/bd6102s/workspace/experiment/drct/gkd_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        else:
                            raise ValueError(f"Unknown Model Type: {model_type}")
                    elif ds == 'gkd_large':
                        data_dir = f'/europa/hpc-homes/bd6102s/workspace/gkd_large/{class_name}/train/HR_{img_resolution}'
                        if model_type == 'drn-l':
                            save = f'/europa/hpc-homes/bd6102s/workspace/experiment/drn-l/gkd_large_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        elif model_type == 'drct':
                            save = f'/europa/hpc-homes/bd6102s/workspace/experiment/drct/gkd_large_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        else:
                            raise ValueError(f"Unknown Model Type: {model_type}")
                    else:
                        raise ValueError(f"Unknown dataset: {ds}")
                else:
                    date_string = now.strftime("%H:%M:%S")
                    if ds == 'mvtec':
                        # point to prepared local dataset
                        data_dir = f'data/mvtec_128/{class_name}/train/good'
                        if model_type == 'drn-l':
                            save = f'./workspace/experiment/drn-l/mvtec_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        elif model_type == 'drct':
                            save = f'./workspace/experiment/drct/mvtec_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        else:
                            raise ValueError(f"Unknown Model Type: {model_type}")
                    elif ds == 'gkd':
                        data_dir = f'workspace/gkd/{class_name}/train/HR_{img_resolution}'
                        if model_type == 'drn-l':
                            save = f'./workspace/experiment/drn-l/gkd_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        elif model_type == 'drct':
                            save = f'./workspace/experiment/drct/gkd_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        else:
                            raise ValueError(f"Unknown Model Type: {model_type}")
                    elif ds == 'gkd_large':
                        data_dir = f'workspace/gkd_large/{class_name}/train/HR_{img_resolution}'
                        if model_type == 'drn-l':
                            save = f'./workspace/experiment/drn-l/gkd_large_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        elif model_type == 'drct':
                            save = f'./workspace/experiment/drct/gkd_large_{class_name}_{img_resolution}_X{scale}{date_string}/'
                        else:
                            raise ValueError(f"Unknown Model Type: {model_type}")
                    else:
                        raise ValueError(f"Unknown dataset: {ds}")

                # MVTec Carpet Datarange
                # data_range: str='1-224/225-280' # train test data range
                if class_name == "carpet":
                    data_range: str='1-224/225-280'
                    dataset_length = 256
                elif class_name == "grid":
                    data_range: str='1-210/211-264'
                    dataset_length = 256
                elif class_name == 'DC0' or class_name == 'DC2':
                    if ds == 'gkd':
                        data_range: str='1-2083/2084-2604' # train test data range
                        # data_range: str='1-12600/12601-14000' # train test data range
                        dataset_length = 2084
                    elif ds == 'gkd_large':
                        data_range: str='1-12600/12601-14000' # train test data range
                        dataset_length = 14000
                    else:
                        raise ValueError(f"Unknown dataset: {ds}")
                else:
                    raise ValueError(f"Unknown class name: {class_name}")

                test_every = dataset_length // batch_size
                print_every = test_every
                patience = 1
                min_delta = 0.005

                n_threads = 4
                loss = '1*L1'

                if model_type == 'drn-l':
                    # DRN-L
                    if pre_train:
                        if slurm:
                            pre_trained = f'/europa/hpc-homes/bd6102s/workspace/pretrained_model_weights/DRNL{scale}x.pt'
                            pre_trained_dual = f'/europa/hpc-homes/bd6102s/workspace/pretrained_model_weights/DRNL{scale}x_dual_model.pt'
                        else:
                            pre_trained = f'workspace/pretrained_model_weights/DRNL{scale}x.pt'
                            pre_trained_dual = f'workspace/pretrained_model_weights/DRNL{scale}x_dual_model.pt'
                    else:
                        pre_trained = '.'
                        pre_trained_dual = '.'
                    opt_drn = DRN()
                    opt_drn = setup_opt_drn(opt_drn, best_auc, ssim_window_size, ds, class_name, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, pre_trained, pre_trained_dual, loss)
                    train_drn(opt_drn)
                elif model_type == 'drct':
                    # DRCT
                    if pre_train:
                        if slurm:
                            pre_trained = '/europa/hpc-homes/bd6102s/workspace/pretrained_model_weights/net_g_latest.pth'
                        else:
                            pre_trained = 'workspace/pretrained_model_weights/net_g_latest.pth'
                    else:
                        pre_trained = '.'
                    opt_drct = DRCT()
                    opt_drct = setup_opt_drct(opt_drct, best_auc, ssim_window_size, ds, class_name, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, img_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, pre_trained, loss)
                    train_drct(opt_drct)
                else:
                    raise ValueError(f"Unknown Model Type: {model_type}")