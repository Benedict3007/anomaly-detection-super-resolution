"""
LEGACY/EXPERIMENTAL MODULE

Utility used for early evaluation and specificity calculations during research.
Not required by the main SR pipeline. Kept for reproducibility of thesis results.
"""

import torch
import math
import matplotlib
import random
import numpy as np
import datetime
import os
import time
import matplotlib.pyplot as plt
import copy
import imageio.v2 as imageio
import logging
import re
from collections import Counter
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
from skimage import exposure

from checkpoint import Checkpoint
from model import Model
from data import Data
from loss import Loss
from trainer import Trainer
from helpers import prepare, quantize, setup_opt_drn, setup_opt_drct, setup_logger, calculate_ssim, calculate_mse, calculate_psnr, min_max_scaling, histogram_equalization, analyze_window_sizes, analyze_window_sizes_gkd, process_images, process_gkd_images, plot_roc_curve, find_threshold_for_perfect_recall

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
    no_augment: bool=True # do not use data augmentation
    pre_train: str='.' # pre-trained model directory
    pre_train_dual: str='.' # pre-trained dual model directory
    n_blocks: int=40 # number of residual blocks, 16|30|40|80
    n_feats: int=20 # number of feature maps 
    negval: float=0.2 # Negative value parameter for Leaky ReLU
    test_every: int=10 # do test per every N batches
    epochs: int=10 # number of epochs to train
    batch_size: int=4 # input batch_size for training
    self_ensemble: bool=False # use self-ensemble method for test
    test_only: bool=True # set this option to test the model
    lr: float=1e-4 # learning rate
    eta_min: float=1e-7 # eta_min learning rate
    beta1: float=0.9 # ADAM beta1
    beta2: float=0.999 # ADAM beta2
    epsilon: float=1e-8 # ADAM epsilon for numerical stability
    weight_decay: float=1e-8 # weight decay
    loss: str='1*L1' # loss function configuration, L1/MSE
    skip_threshold: float=1e6 # skipping batch that has large error
    dual_weight: float=0.1 # the weight of dual loss
    save: str='./workspace/experiment/drn-l/gkd_dc2_unlabeld_X4_10_grayscale/' # file name to save
    print_every: int=10 # how many batches to wait before logging training status
    save_results: bool=False # save output results
    dual: bool=True
    patience: int=10

@dataclass
class DRCT:
    model_name: str='drct'
    n_threads: int=2 # number of threads for data loading
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
    no_augment: bool=True # do not use data augmentation
    pre_train: str='.' # pre-trained model directory
    pre_train_dual: str='.' # pre-trained dual model directory
    negval: float=0.2 # Negative value parameter for Leaky ReLU
    test_every: int=30 # do test per every N batches
    epochs: int=10 # number of epochs to train
    batch_size: int=2 # input batch_size for training
    self_ensemble: bool=False # use self-ensemble method for test
    test_only: bool=True # set this option to test the model
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
    save_results: bool=False # save output results
    dual: bool=True
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

def calc_specificity(model_type, model_name):
    model_type = model_type
    model_name = model_name

    # Extract dataset
    dataset = model_name.split('_')[0]

    # Extract class and resolution
    if dataset == 'mvtec':
        mvtec_class = model_name.split('_')[2]
        gkd_class = None
    elif dataset == 'gkd':
        mvtec_class = None
        gkd_class = model_name.split('_')[2]
    else:
        mvtec_class = None
        gkd_class = None

    resolution = int(re.search(r'_(\d{2,3})_', model_name).group(1))

    scaling = int(re.search(r'X(\d)', model_name).group(1))

    # Print results
    # print(f"dataset: {dataset}, mvtec_class: {mvtec_class}, gkd_class: {gkd_class}, resolution: {resolution}, scaling: {scaling}")
    # Parameter
    scale = scaling
    no_augment = False
    epochs = 1000
    batch_size = 1 
    patch_size = resolution
    img_size = resolution // scale

    if dataset == 'mvtec' and mvtec_class == 'carpet':
        n_colors = 3
    else:
        n_colors = 1
    # n_colors = 3

    if dataset == 'mvtec':
        data_dir_good = f'./workspace/mvtec_anomaly_detection_modified/{mvtec_class}/test/HR_{resolution}/good'
        data_dir_bad = f'./workspace/mvtec_anomaly_detection_modified/{mvtec_class}/test/HR_{resolution}/bad'
        dataset_length = 256
    elif dataset == 'gkd':
        data_dir_good = f'./workspace/gkd/{gkd_class}/test/HR_{resolution}/good'
        data_dir_bad = f'./workspace/gkd/{gkd_class}/test/HR_{resolution}/bad'
        dataset_length = 2048
    else:
        print("Not the right dataset!")

    now = datetime.datetime.now()

    # data_range: str='1-800/801-1000'
    # MVTec Carpet Datarange
    data_range: str='' # train test data range
    test_every = dataset_length // batch_size
    print_every = test_every
    patience = 1000
    n_threads = 4

    # if model_type == 'drn-l':
    #     save = f'./workspace/images/drn-l/{model_name}/'
    #     opt_drn = DRN()
    #     opt_good = copy.deepcopy(opt_drn)
    #     opt_bad = copy.deepcopy(opt_drn)
    #     opt_good = setup_opt_drn(opt_good, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir_good, save, data_range, test_every, print_every, patience, n_threads, model_name)
    #     opt_bad = setup_opt_drn(opt_bad, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir_bad, save, data_range, test_every, print_every, patience, n_threads, model_name)
    # elif model_type == 'drct':
    #     save = f'./workspace/images/drct/{model_name}/'
    #     opt_drct = DRCT()
    #     opt_good = copy.deepcopy(opt_drct)
    #     opt_bad = copy.deepcopy(opt_drct)
    #     opt_good = setup_opt_drct(opt_good, scale, no_augment, n_colors, epochs, batch_size, patch_size, img_size, data_dir_good, save, data_range, test_every, print_every, patience, n_threads, model_name)
    #     opt_bad = setup_opt_drct(opt_bad, scale, no_augment, n_colors, epochs, batch_size, patch_size, img_size, data_dir_bad, save, data_range, test_every, print_every, patience, n_threads, model_name)
    # else:
    #     print("Model_Type unknown!")
        
    # if model_type == 'drn-l':
    #     checkpoint = Checkpoint(opt_good)
    #     model = Model(opt_good, checkpoint, dual_model=True)
    #     model.eval()
        
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #     loader_good = Data(opt_good)
    #     loader_bad = Data(opt_bad)
        
    #     test_loader_good = loader_good.loader_test
    #     test_loader_bad = loader_bad.loader_test
    #     filepath_good = f"workspace/images/drn-l/{model_name}/predicted_images/good"
    #     filepath_bad = f"workspace/images/drn-l/{model_name}/predicted_images/bad"
    #     os.makedirs(filepath_good, exist_ok=True)
    #     os.makedirs(filepath_bad, exist_ok=True)
    # elif model_type == 'drct':
    #     checkpoint = Checkpoint(opt_good)
    #     model = Model(opt_good, checkpoint, dual_model=False)
    #     model.eval()
        
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #     loader_good = Data(opt_good)
    #     loader_bad = Data(opt_bad)
        
    #     test_loader_good = loader_good.loader_test
    #     test_loader_bad = loader_bad.loader_test
    #     filepath_good = f"workspace/images/drct/{model_name}/predicted_images/good"
    #     filepath_bad = f"workspace/images/drct/{model_name}/predicted_images/bad"
    #     os.makedirs(filepath_good, exist_ok=True)
    #     os.makedirs(filepath_bad, exist_ok=True)
    # else:
    #     print("Model_Type unknown!")

    # with torch.no_grad():
    #     scale = opt_good.scale
    #     for si, s in enumerate([scale]):
    #         eval_psnr = 0
    #         for _, (lr, hr, filename) in enumerate(test_loader_good):
    #             filename = filename[0]
    #             no_eval = (hr.nelement() == 1)
    #             if not no_eval:
    #                 lr, hr = prepare(device, lr, hr)
    #             else:
    #                 lr, = prepare(device, lr)

    #             sr = model(lr[0])
                
    #             if isinstance(sr, list): sr = sr[-1]
                
    #             sr = quantize(sr, opt_good.rgb_range)
                                                
    #             # Save super-resolved image
    #             filename = os.path.join(filepath_good, filename)
    #             normalized = sr[0].data.mul(255 / opt_good.rgb_range)
            
    #             ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            
    #             # imageio.imwrite('{}.png'.format(filename), ndarr)
    #             # Convert to PIL Image and save
    #             if ndarr.ndim == 2:
    #                 img = Image.fromarray(ndarr, mode='L')
    #             elif ndarr.ndim == 3:
    #                 if ndarr.shape[2] == 1:
    #                     img = Image.fromarray(ndarr.squeeze(), mode='L')
    #                 elif ndarr.shape[2] == 3:
    #                     img = Image.fromarray(ndarr, mode='RGB')
    #                 elif ndarr.shape[2] == 4:
    #                     img = Image.fromarray(ndarr, mode='RGBA')
    #                 else:
    #                     raise ValueError(f"Unexpected number of channels: {ndarr.shape[2]}")
    #             else:
    #                 raise ValueError(f"Unexpected number of dimensions: {ndarr.ndim}")
                
    #             img.save(f'{filename}.png')

    # print("Process completed.")  

    # with torch.no_grad():
    #     scale = opt_bad.scale
    #     for si, s in enumerate([scale]):
    #         eval_psnr = 0
    #         for _, (lr, hr, filename) in enumerate(test_loader_bad):
    #             filename = filename[0]
    #             no_eval = (hr.nelement() == 1)
    #             if not no_eval:
    #                 lr, hr = prepare(device, lr, hr)
    #             else:
    #                 lr, = prepare(device, lr)

    #             # Super-resolution model prediction
    #             sr = model(lr[0])
    #             if isinstance(sr, list): sr = sr[-1]
    #             sr = quantize(sr, opt_bad.rgb_range)
                
    #             # Save super-resolved image
    #             filename = os.path.join(filepath_bad, filename)
    #             normalized = sr[0].data.mul(255 / opt_bad.rgb_range)
    #             ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    #             # imageio.imwrite('{}.png'.format(filename), ndarr)
    #             if ndarr.ndim == 2:
    #                 img = Image.fromarray(ndarr, mode='L')
    #             elif ndarr.ndim == 3:
    #                 if ndarr.shape[2] == 1:
    #                     img = Image.fromarray(ndarr.squeeze(), mode='L')
    #                 elif ndarr.shape[2] == 3:
    #                     img = Image.fromarray(ndarr, mode='RGB')
    #                 elif ndarr.shape[2] == 4:
    #                     img = Image.fromarray(ndarr, mode='RGBA')
    #                 else:
    #                     raise ValueError(f"Unexpected number of channels: {ndarr.shape[2]}")
    #             else:
    #                 raise ValueError(f"Unexpected number of dimensions: {ndarr.ndim}")
                
    #             img.save(f'{filename}.png')


    # print("Process completed.")   

    if dataset == 'mvtec':
        if model_type == 'drn-l':
            good_original_folder = f"workspace/mvtec_anomaly_detection_modified/{mvtec_class}/test/HR_{resolution}/good/HR"
            bad_original_folder = f"workspace/mvtec_anomaly_detection_modified/{mvtec_class}/test/HR_{resolution}/bad/HR"
            good_reconstructed_folder = f"workspace/images/drn-l/{model_name}/predicted_images/good"
            bad_reconstructed_folder = f"workspace/images/drn-l/{model_name}/predicted_images/bad"
            log_file_path = f"workspace/images/drn-l/{model_name}/scores.txt"
        elif model_type == 'drct':
            good_original_folder = f"workspace/mvtec_anomaly_detection_modified/{mvtec_class}/test/HR_{resolution}/good/HR"
            bad_original_folder = f"workspace/mvtec_anomaly_detection_modified/{mvtec_class}/test/HR_{resolution}/bad/HR"
            good_reconstructed_folder = f"workspace/images/drct/{model_name}/predicted_images/good"
            bad_reconstructed_folder = f"workspace/images/drct/{model_name}/predicted_images/bad"
            log_file_path = f"workspace/images/drct/{model_name}/scores.txt"
        else:
            print("Unknown Model Type")
    elif dataset == 'gkd':
        if model_type == 'drn-l':
            good_original_folder = f"workspace/gkd/{gkd_class}/test/HR_{resolution}/good/HR"
            bad_original_folder = f"workspace/gkd/{gkd_class}/test/HR_{resolution}/bad/HR"
            good_reconstructed_folder = f"workspace/images/drn-l/{model_name}/predicted_images/good"
            bad_reconstructed_folder = f"workspace/images/drn-l/{model_name}/predicted_images/bad"
            log_file_path = f"workspace/images/drn-l/{model_name}/scores.txt"
        elif model_type == 'drct':
            good_original_folder = f"workspace/gkd/{gkd_class}/test/HR_{resolution}/good/HR"
            bad_original_folder = f"workspace/gkd/{gkd_class}/test/HR_{resolution}/bad/HR"
            good_reconstructed_folder = f"workspace/images/drct/{model_name}/predicted_images/good"
            bad_reconstructed_folder = f"workspace/images/drct/{model_name}/predicted_images/bad"
            log_file_path = f"workspace/images/drct/{model_name}/scores.txt"
        else:
            print("Unknown Model Type")
    else: 
        print("Unknown Dataset!")

    if dataset == 'mvtec':
        analysis_results = analyze_window_sizes(good_original_folder, good_reconstructed_folder,
                                                bad_original_folder, bad_reconstructed_folder)

        # print(f"Best window size (max difference): {analysis_results['best_window_size']}")
        # print(f"Max difference in SSIM scores: {analysis_results['max_difference']:.4f}")
        # print(f"Best window size (max AUC): {analysis_results['best_auc_window_size']}")
        # print(f"Max AUC: {analysis_results['max_auc']:.4f}")
        # Choose which window size to use (max difference or max AUC)
        chosen_window_size = analysis_results['best_auc_window_size']

        y_true, y_scores_ssim, y_scores_mse, y_scores_psnr = process_images(
            good_original_folder, good_reconstructed_folder,
            bad_original_folder, bad_reconstructed_folder,
            log_file_path, chosen_window_size
        )

        r_a_ssim = roc_auc_score(y_true, y_scores_ssim)
        r_a_mse = roc_auc_score(y_true, y_scores_mse)
        r_a_psnr = roc_auc_score(y_true, y_scores_psnr)
        # print(f"Number of samples: {len(y_true)}")
        # print(f"Number of anomalies: {sum(y_true)}")
        # print(f"Number of normal samples: {len(y_true) - sum(y_true)}")
        # print(f"AUC-ROC (SSIM): {r_a_ssim:.4f}")
        # print(f"AUC-ROC (MSE): {r_a_mse:.4f}")
        # print(f"AUC-ROC (PSNR): {r_a_psnr:.4f}")
    elif dataset == 'gkd':
        analysis_results = analyze_window_sizes_gkd(good_original_folder, good_reconstructed_folder,
                                                bad_original_folder, bad_reconstructed_folder)

        # print(f"Best window size (max difference): {analysis_results['best_window_size']}")
        # print(f"Max difference in SSIM scores: {analysis_results['max_difference']:.4f}")
        # print(f"Best window size (max AUC): {analysis_results['best_auc_window_size']}")
        # print(f"Max AUC: {analysis_results['max_auc']:.4f}")
        # Choose which window size to use (max difference or max AUC)
        chosen_window_size = analysis_results['best_auc_window_size']

        y_true, y_scores_ssim, y_scores_mse, y_scores_psnr = process_gkd_images(
            good_original_folder, good_reconstructed_folder,
            bad_original_folder, bad_reconstructed_folder,
            log_file_path, chosen_window_size
        )
        r_a_ssim = roc_auc_score(y_true, y_scores_ssim)
        r_a_mse = roc_auc_score(y_true, y_scores_mse)
        r_a_psnr = roc_auc_score(y_true, y_scores_psnr)
        # print(f"Number of samples: {len(y_true)}")
        # print(f"Number of anomalies: {sum(y_true)}")
        # print(f"Number of normal samples: {len(y_true) - sum(y_true)}")
        # print(f"AUC-ROC (SSIM): {r_a_ssim:.4f}")
        # print(f"AUC-ROC (MSE): {r_a_mse:.4f}")
        # print(f"AUC-ROC (PSNR): {r_a_psnr:.4f}")
    else: 
        print("Unknown Dataset!")    

    def specificity(y_true, y_scores):
        tn, fp, fn, tp = confusion_matrix(y_true, y_scores).ravel()
        return tn / (tn + fp)

    optimal_threshold_ssim = find_threshold_for_perfect_recall(y_true, y_scores_ssim)
    predictions_ssim = (y_scores_ssim >= optimal_threshold_ssim).astype(int)
    specificity_ssim = specificity(y_true, predictions_ssim)

    optimal_threshold_mse = find_threshold_for_perfect_recall(y_true, y_scores_mse)
    predictions_mse = (y_scores_mse >= optimal_threshold_mse).astype(int)
    specificity_mse = specificity(y_true, predictions_mse)

    optimal_threshold_psnr = find_threshold_for_perfect_recall(y_true, y_scores_psnr)
    predictions_psnr = (y_scores_psnr >= optimal_threshold_psnr).astype(int)
    specificity_psnr = specificity(y_true, predictions_psnr)

    return f'SSIM: {specificity_ssim:.2f}, MSE: {specificity_mse:.2f}, PSNR: {specificity_psnr:.2f}, Model Name: {model_name}'

if __name__ == "__main__":
    models = ['gkd_large_DC0_128_X810:02:32',
                'gkd_large_DC2_128_X823:04:06']

    model_type = 'drct'
    
    specificity_scores = []

    for model in models:
        print(f"Current Model: {model}")
        score = calc_specificity(model_type, model)
        specificity_scores.append(score)

    print(specificity_scores)

