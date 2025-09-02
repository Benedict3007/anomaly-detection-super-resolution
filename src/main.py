import torch
import matplotlib
import random
import numpy as np
import datetime
import time
import copy
import argparse
import sys
import os
import yaml
from dataclasses import dataclass
from typing import List, Optional
from sklearn.metrics import roc_auc_score
from src.metrics import ssim_numpy as calculate_ssim, psnr_numpy as calculate_psnr


from src.checkpoint import Checkpoint
from src.model import Model
from src.data import Data
from src.loss import Loss
from src.trainer import Trainer

matplotlib.use('Agg')

def set_seed(seed: int) -> None:
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
    scale: int | list[int]=4 # super resolution scale
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
    min_delta: float=0.0
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
    scale: int | list[int]=4 # super resolution scale
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
    depths: tuple[int, ...]=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    embed_dim: int=180
    num_heads: tuple[int, ...]=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    mlp_ratio: int=2
    upsampler: str='pixelshuffle'
    resi_connection: str='1conv'
    ema_decay: float=0.999
    weight_decay: float=0.0
    betas: tuple[float, float]=(0.9, 0.99)
    patience: int=10
    min_delta: float=0.0
    dataset: str=''
    classe: str=''
    slurm: bool=False
    ssim_window_size: int=11
    best_auc: float=1.0

def setup_opt_drn(
    opt: DRN,
    best_auc: float,
    ssim_window_size: int,
    dataset: str,
    classe: str,
    slurm: bool,
    scale: int,
    no_augment: bool,
    n_colors: int,
    epochs: int,
    batch_size: int,
    patch_size: int,
    data_dir: str,
    save: str,
    data_range: str,
    test_every: int,
    print_every: int,
    patience: int,
    min_delta: float,
    n_threads: int,
    pre_trained: str,
    pre_trained_dual: str,
    loss: str,
) -> DRN:
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

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # First, parse only --config if provided
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    # Main parser with all options
    parser = argparse.ArgumentParser(description='Training/Evaluation entrypoint', parents=[pre_parser])
    parser.add_argument('--model-type', type=str, default='drct', choices=['drct', 'drn-l'])
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec'])
    parser.add_argument('--classe', type=str, default='grid', choices=['grid', 'carpet'])
    parser.add_argument('--scale', type=int, default=4, choices=[4, 8])
    parser.add_argument('--resolution', type=int, default=128, choices=[32, 64, 128, 256])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--data-root', type=str, default='auto')
    parser.add_argument('--save-dir', type=str, default='./workspace/experiment')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    # DataLoader workers (default 0 on macOS to avoid runpy RuntimeWarning spam)
    default_workers = 0 if sys.platform == 'darwin' else 4
    parser.add_argument('--workers', type=int, default=default_workers)

    # If a config file was provided, load it and set defaults before final parse
    if pre_args.config is not None and os.path.isfile(pre_args.config):
        with open(pre_args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        # Normalize keys: prefer underscores in argparse dest
        normalized_cfg = {k.replace('-', '_'): v for k, v in cfg.items()}
        parser.set_defaults(**normalized_cfg)

    return parser.parse_args(argv)

def setup_opt_drct(
    opt: DRCT,
    best_auc: float,
    ssim_window_size: int,
    dataset: str,
    classe: str,
    slurm: bool,
    scale: int,
    no_augment: bool,
    n_colors: int,
    epochs: int,
    batch_size: int,
    patch_size: int,
    img_size: int,
    data_dir: str,
    save: str,
    data_range: str,
    test_every: int,
    print_every: int,
    patience: int,
    min_delta: float,
    n_threads: int,
    pre_trained: str,
    loss: str,
) -> DRCT:
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
    
def train_drn(opt_drn: DRN) -> None:
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
    

        print("Training completed")
        end_time = time.time()
        checkpoint_drn.write_log(f"Total Training Time: {((end_time -  start_time)/3600):.2f}")
        
        # Post-training evaluation on mvtec_128 val/good (PSNR/SSIM)
        try:
            eval_opt = copy.deepcopy(opt_drn)
            eval_opt.test_only = True
            eval_opt.no_augment = True
            eval_opt.batch_size = 1
            eval_opt.data_dir = f'data/mvtec_128/{opt_drn.classe}/val/good'
            eval_opt.data_test = 'mvtec_val_good'
            eval_loader = Data(eval_opt)
            t.loader_test = eval_loader.loader_test
            t.test()
        except Exception as e:
            print(f"Evaluation skipped due to error: {e}")

        # Skip anomaly AUC on validation since val contains only good images
        checkpoint_drn.write_log("Skipping anomaly AUC on validation (good-only split)")

        checkpoint_drn.save(t, opt_drn.epochs, is_best=True, dual_model=True)
        checkpoint_drn.done()

def train_drct(opt_drct: DRCT) -> None:
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

        # Post-training evaluation on mvtec_128 val/good (PSNR/SSIM)
        try:
            eval_opt = copy.deepcopy(opt_drct)
            eval_opt.test_only = True
            eval_opt.no_augment = True
            eval_opt.batch_size = 1
            eval_opt.data_dir = f'data/mvtec_128/{opt_drct.classe}/val/good'
            eval_opt.data_test = 'mvtec_val_good'
            eval_loader = Data(eval_opt)
            t.loader_test = eval_loader.loader_test
            t.test()
        except Exception as e:
            print(f"Evaluation skipped due to error: {e}")

        # Skip anomaly AUC on validation since val contains only good images
        checkpoint_drct.write_log("Skipping anomaly AUC on validation (good-only split)")

        # Disk-based evaluation removed per request

        checkpoint_drct.save(t, opt_drct.epochs, is_best=True, dual_model=False)
        checkpoint_drct.done()

if __name__ == "__main__":
    args = parse_args()
    slurm = False
    # Defaults for local runs (used by evaluation utilities)
    best_auc = 0.0
    ssim_window_size = 11

    model_type = args.model_type
    pre_train = args.pretrain
    ds = args.dataset
    class_name = args.classe
    img_resolution = args.resolution
    scale = args.scale
    epochs = args.epochs
    batch_size = args.batch_size
    no_augment = args.no_augment

    print(f"Model: {model_type}")
    print(f"Dataset: {ds}")
    print(f"Class: {class_name}")
    print(f"Resolution: {img_resolution}")
    print(f"Scale: {scale}")

    # Set channels based on class: carpet=RGB(3), grid/grayscale=1
    n_colors = 3 if (ds == 'mvtec' and class_name == 'carpet') else 1

    patch_size = img_resolution
    img_size = img_resolution // scale

    now = datetime.datetime.now()
    date_string = now.strftime("%H:%M:%S")

    # Data/save paths (simple local defaults)
    if ds == 'mvtec':
        data_root = args.data_root
        if data_root == 'auto':
            data_root = f"data/mvtec_{img_resolution}"
        data_dir = f"{data_root}/{class_name}/train/good"
    elif ds == 'gkd':
        data_dir = f"workspace/gkd/{class_name}/train/HR_{img_resolution}"
    elif ds == 'gkd_large':
        data_dir = f"workspace/gkd_large/{class_name}/train/HR_{img_resolution}"
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    save = f"{args.save_dir}/{model_type}/mvtec_{class_name}_{img_resolution}_X{scale}{date_string}/" if ds == 'mvtec' else f"{args.save_dir}/{model_type}/{ds}_{class_name}_{img_resolution}_X{scale}{date_string}/"

    # Datarange and dataset length (kept simple)
    if ds == 'mvtec':
        data_range = '1-210/211-264' if class_name == 'grid' else '1-224/225-280'
        dataset_length = 256
    elif ds == 'gkd':
        data_range = '1-2083/2084-2604'
        dataset_length = 2084
    else:
        data_range = ''
        dataset_length = batch_size

    test_every = dataset_length // batch_size
    print_every = test_every
    patience = 1
    min_delta = 0.005
    n_threads = 4
    loss = '1*L1'

    if model_type == 'drn-l':
        if pre_train:
            pre_trained = f'workspace/pretrained_model_weights/DRNL{scale}x.pt'
            pre_trained_dual = f'workspace/pretrained_model_weights/DRNL{scale}x_dual_model.pt'
        else:
            pre_trained = '.'
            pre_trained_dual = '.'
        opt_drn = DRN()
        opt_drn = setup_opt_drn(opt_drn, best_auc, ssim_window_size, ds, class_name, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, pre_trained, pre_trained_dual, loss)
        if args.device == 'cpu':
            opt_drn.cpu = True
        train_drn(opt_drn)
    elif model_type == 'drct':
        pre_trained = 'workspace/pretrained_model_weights/net_g_latest.pth' if pre_train else '.'
        opt_drct = DRCT()
        opt_drct = setup_opt_drct(opt_drct, best_auc, ssim_window_size, ds, class_name, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, img_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, pre_trained, loss)
        opt_drct.cpu = (args.device == 'cpu')
        opt_drct.test_only = args.test_only
        train_drct(opt_drct)
    else:
        raise ValueError(f"Unknown Model Type: {model_type}")