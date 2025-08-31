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
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
from skimage import exposure

from src.checkpoint import Checkpoint
from src.model import Model
from src.data import Data
from src.loss import Loss
# from trainer import Trainer

def prepare(device, *args):
    device = device

    if len(args)>1:
        return [a.to(device) for a in args[0]], args[-1].to(device)
    return [a.to(device) for a in args[0]],

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def setup_opt_drn(opt, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir, save, data_range, test_every, print_every, patience, n_threads, model_name):
    opt.scale = scale
    opt.scale = [pow(2, s+1) for s in range(int(np.log2(opt.scale)))] 

    if scale == 2:
        opt.n_blocks = 44
        opt.n_feats = 40
    elif scale == 4:
        opt.n_blocks = 40
        opt.n_feats = 20
        # opt.pre_train = './workspace/pretrained_model_weights/DRNL4x.pt'
        # opt.pre_train_dual = './workspace/pretrained_model_weights/DRNL4x_dual_model.pt'
    elif scale == 8:
        opt.n_blocks = 36
        opt.n_feats = 10
        # opt.pre_train = './workspace/pretrained_model_weights/DRNL8x.pt'
        # opt.pre_train_dual = './workspace/pretrained_model_weights/DRNL8x_dual_model.pt'
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
    opt.n_threads = n_threads
    opt.pre_train = f'./workspace/experiment/drn-l/{model_name}/model/model_best.pt'
    opt.pre_train_dual = f'./workspace/experiment/drn-l/{model_name}/model/dual_model_best.pt'
    

    return opt

def setup_opt_drct(opt, scale, no_augment, n_colors, epochs, batch_size, patch_size, img_size, data_dir, save, data_range, test_every, print_every, patience, n_threads, model_name):
    opt.scale = scale
    opt.upscale = scale
    opt.scale = [opt.scale] 
    opt.no_augment = no_augment
    opt.n_colors = n_colors
    opt.epochs = epochs
    opt.batch_size = batch_size
    opt.patch_size = patch_size
    opt.data_dir = data_dir
    opt.save = save
    opt.test_every = test_every
    opt.print_every = print_every
    opt.img_size = img_size
    opt.patience = patience
    opt.n_threads = n_threads
    opt.pre_train = f'./workspace/experiment/drct/{model_name}/model/model_best.pt'
    # opt.pre_train = './workspace/pretrained_model_weights/net_g_latest.pth'
    opt.window_size = img_size // 4
    
    return opt

def setup_logger(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')

def calculate_ssim(original, reconstructed, win_size):
    # Determine data range based on dtype; assume float images are in [0,1]
    dr = 1.0 if np.issubdtype(np.asarray(original).dtype, np.floating) else 255
    if len(original.shape) == 2 and len(reconstructed.shape) == 2:
        return ssim(original, reconstructed, win_size=win_size, data_range=dr)
    elif len(original.shape) == 3 and len(reconstructed.shape) == 3:
        return ssim(original, reconstructed, win_size=win_size, data_range=dr, channel_axis=-1)
    else:
        raise ValueError("Input images must have the same dimensions (both 2D or both 3D)")
        
def calculate_mse(original, reconstructed):
    o = np.asarray(original, dtype=np.float32)
    r = np.asarray(reconstructed, dtype=np.float32)
    return float(np.mean((o - r) ** 2))

def calculate_psnr(original, reconstructed):
    dr = 1.0 if np.issubdtype(np.asarray(original).dtype, np.floating) else 255
    return psnr(original, reconstructed, data_range=dr)

def min_max_scaling(image_array):
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    return ((image_array - min_val) * (255 / (max_val - min_val))).astype(np.uint8)

def histogram_equalization(image_array):
    # Ensure the input array is in float format
    image_array = image_array.astype(float)
    
    # Normalize to 0-1 range
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
    # Apply histogram equalization
    if len(image_array.shape) == 2:  # Grayscale image
        eq_array = exposure.equalize_hist(image_array)
    elif len(image_array.shape) == 3:  # Color image
        eq_array = np.dstack([exposure.equalize_hist(image_array[:,:,i]) 
                              for i in range(image_array.shape[2])])
    
    # Convert back to 0-255 range
    return (eq_array * 255).astype(np.uint8)

def analyze_window_sizes(good_original_folder, good_reconstructed_folder, 
                         bad_original_folder, bad_reconstructed_folder, 
                         min_size=3, max_size=None, step=10):
    # def load_image(image_path):
    #     return np.array(Image.open(image_path))
    
    def load_image(image_path):
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)

    def process_folder(folder_original, folder_reconstructed):
        results = []
        for img_name in os.listdir(folder_original):
            original_path = os.path.join(folder_original, img_name)
            reconstructed_path = os.path.join(folder_reconstructed, img_name)
            
            original = load_image(original_path)
            # original = min_max_scaling(original)
            # original = histogram_equalization(original)
            # print(original.shape)
            reconstructed = load_image(reconstructed_path)
            # reconstructed = min_max_scaling(reconstructed)
            # reconstructed = histogram_equalization(reconstructed)
            
            min_dim = min(original.shape[0], original.shape[1])
            actual_max_size = min(max_size, min_dim - 3) if max_size else min_dim - 3
            actual_max_size = actual_max_size if actual_max_size % 2 != 0 else actual_max_size - 1

            image_results = []
            for win_size in range(min_size, actual_max_size + 1, step):
                ssim_score = calculate_ssim(original, reconstructed, win_size)
                image_results.append(ssim_score)
            results.append(image_results)
        return results, actual_max_size

    good_results, good_max_size = process_folder(good_original_folder, good_reconstructed_folder)
    bad_results, bad_max_size = process_folder(bad_original_folder, bad_reconstructed_folder)

    # Use the smaller of the two max sizes
    actual_max_size = min(good_max_size, bad_max_size)

    window_sizes = list(range(min_size, actual_max_size + 1, step))
    avg_good_scores = np.mean(good_results, axis=0)
    avg_bad_scores = np.mean(bad_results, axis=0)
    score_differences = avg_good_scores - avg_bad_scores

    best_window_size = window_sizes[np.argmax(score_differences)]
    max_difference = np.max(score_differences)

    auc_scores = []
    for i in range(len(window_sizes)):
        y_true = [0] * len(good_results) + [1] * len(bad_results)
        y_scores = [1 - score[i] for score in good_results] + [1 - score[i] for score in bad_results]
        auc_scores.append(roc_auc_score(y_true, y_scores))

    best_auc_window_size = window_sizes[np.argmax(auc_scores)]
    max_auc = np.max(auc_scores)

    return {
        'window_sizes': window_sizes,
        'avg_good_scores': avg_good_scores.tolist(),
        'avg_bad_scores': avg_bad_scores.tolist(),
        'score_differences': score_differences.tolist(),
        'best_window_size': best_window_size,
        'max_difference': max_difference,
        'auc_scores': auc_scores,
        'best_auc_window_size': best_auc_window_size,
        'max_auc': max_auc
    }

def analyze_window_sizes_gkd(good_original_folder, good_reconstructed_folder, 
                             bad_original_folder, bad_reconstructed_folder, 
                             min_size=3, max_size=None, step=10):
    # def load_image(image_path):
    #     return np.array(Image.open(image_path))
    def load_image(image_path):
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    
    def process_folder(folder_original, folder_reconstructed):
        results = defaultdict(lambda: defaultdict(list))
        max_window_size = float('inf')
        
        for img_name in os.listdir(folder_original):
            original_path = os.path.join(folder_original, img_name)
            reconstructed_path = os.path.join(folder_reconstructed, img_name)
            
            original = load_image(original_path)
            # original = min_max_scaling(original)
            # original = histogram_equalization(original)
            # print(original.shape)
            reconstructed = load_image(reconstructed_path)
            # reconstructed = min_max_scaling(reconstructed)
            # reconstructed = histogram_equalization(reconstructed)
            
            patch_max_size = min(original.shape[0], original.shape[1]) - 3
            patch_max_size = patch_max_size if patch_max_size % 2 != 0 else patch_max_size - 1
            max_window_size = min(max_window_size, patch_max_size)
            
            full_image_id = int(img_name.split('_')[0]) // 14
            
            for win_size in range(min_size, patch_max_size + 1, step):
                ssim_score = calculate_ssim(original, reconstructed, win_size)
                if ssim_score is not None and not np.isnan(ssim_score):
                    results[full_image_id][win_size].append(1 - ssim_score)
        
        return results, max_window_size
    
    good_results, good_max_size = process_folder(good_original_folder, good_reconstructed_folder)
    bad_results, bad_max_size = process_folder(bad_original_folder, bad_reconstructed_folder)
    
    actual_max_size = min(good_max_size, bad_max_size)
    if max_size:
        actual_max_size = min(actual_max_size, max_size)
    
    window_sizes = list(range(min_size, actual_max_size + 1, step))
    
    good_max_scores = {win_size: [max(scores[win_size]) for scores in good_results.values() if win_size in scores] 
                       for win_size in window_sizes}
    bad_max_scores = {win_size: [max(scores[win_size]) for scores in bad_results.values() if win_size in scores] 
                      for win_size in window_sizes}
    
    avg_good_scores = [np.mean(good_max_scores[win_size]) for win_size in window_sizes]
    avg_bad_scores = [np.mean(bad_max_scores[win_size]) for win_size in window_sizes]
    
    score_differences = np.array(avg_good_scores) - np.array(avg_bad_scores)
    
    best_window_size = window_sizes[np.argmax(score_differences)]
    max_difference = np.max(score_differences)
    
    auc_scores = []
    valid_window_sizes = []
    for win_size in window_sizes:
        if len(good_max_scores[win_size]) > 0 and len(bad_max_scores[win_size]) > 0:
            y_true = [0] * len(good_max_scores[win_size]) + [1] * len(bad_max_scores[win_size])
            y_scores = good_max_scores[win_size] + bad_max_scores[win_size]
            auc_scores.append(roc_auc_score(y_true, y_scores))
            valid_window_sizes.append(win_size)
    
    best_auc_window_size = valid_window_sizes[np.argmax(auc_scores)] if auc_scores else None
    max_auc = np.max(auc_scores) if auc_scores else None

    return {
        'window_sizes': window_sizes,
        'avg_good_scores': avg_good_scores,
        'avg_bad_scores': avg_bad_scores,
        'score_differences': score_differences.tolist(),
        'best_window_size': best_window_size,
        'max_difference': max_difference,
        'auc_scores': auc_scores,
        'best_auc_window_size': best_auc_window_size,
        'max_auc': max_auc,
        'valid_window_sizes': valid_window_sizes
    }

def process_images(good_original_folder, good_reconstructed_folder, 
                   bad_original_folder, bad_reconstructed_folder, 
                   log_file_path, window_size):
    setup_logger(log_file_path)
    
    y_true = []
    y_scores_ssim = []
    y_scores_mse = []
    y_scores_psnr = []
    
    # def load_image(image_path):
    #     return np.array(Image.open(image_path))
    def load_image(image_path):
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)

    def process_folder(folder_original, folder_reconstructed, label):
        for img_name in os.listdir(folder_original):
            original_path = os.path.join(folder_original, img_name)
            reconstructed_path = os.path.join(folder_reconstructed, img_name)
            
            original = load_image(original_path)
            # original = min_max_scaling(original)
            # original = histogram_equalization(original)
            # print(original.shape)
            reconstructed = load_image(reconstructed_path)
            # reconstructed = min_max_scaling(reconstructed)
            # reconstructed = histogram_equalization(reconstructed)
            
            ssim_score = calculate_ssim(original, reconstructed, window_size)
            mse_score = calculate_mse(original, reconstructed)
            psnr_score = calculate_psnr(original, reconstructed)
            
            y_true.append(label)
            y_scores_ssim.append(1 - ssim_score)
            y_scores_mse.append(mse_score)
            y_scores_psnr.append(-psnr_score)
            
            logging.info(f"Image: {img_name}, Label: {'Anomalous' if label else 'Normal'}, "
                         f"SSIM (window size {window_size}): {ssim_score:.4f}, "
                         f"MSE: {mse_score:.4f}, PSNR: {psnr_score:.4f}")

    process_folder(good_original_folder, good_reconstructed_folder, 0)
    process_folder(bad_original_folder, bad_reconstructed_folder, 1)
    
    return y_true, y_scores_ssim, y_scores_mse, y_scores_psnr

def process_gkd_images(good_original_folder, good_reconstructed_folder, 
                   bad_original_folder, bad_reconstructed_folder, log_file_path, window_size):
    setup_logger(log_file_path)
    
    y_true = []
    y_scores_ssim = []
    y_scores_mse = []
    y_scores_psnr = []

    # def load_image(image_path):
    #     return np.array(Image.open(image_path))
    
    def load_image(image_path):
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)

    def process_folder(folder_original, folder_reconstructed, label):
        patch_scores = defaultdict(lambda: {'ssim': [], 'mse': [], 'psnr': []})
        
        for img_name in os.listdir(folder_original):
            original_path = os.path.join(folder_original, img_name)
            reconstructed_path = os.path.join(folder_reconstructed, img_name)
            
            original = load_image(original_path)
            # original = min_max_scaling(original)
            # original = histogram_equalization(original)
            # print(original.shape)
            reconstructed = load_image(reconstructed_path)
            # reconstructed = min_max_scaling(reconstructed)
            # reconstructed = histogram_equalization(reconstructed)
            
            ssim_score = calculate_ssim(original, reconstructed, window_size)
            mse_score = calculate_mse(original, reconstructed)
            psnr_score = calculate_psnr(original, reconstructed)
            
            # Group patches by their first number (full image identifier)
            full_image_id = int(img_name.split('_')[0]) // 14
            patch_scores[full_image_id]['ssim'].append(1 - ssim_score)
            patch_scores[full_image_id]['mse'].append(mse_score)
            patch_scores[full_image_id]['psnr'].append(-psnr_score)
            
            # Log the scores
            logging.info(f"Image: {img_name}, Image_Id: {full_image_id}, Label: {'Anomalous' if label else 'Normal'}, "
                         f"SSIM (window size {window_size}): {ssim_score:.4f}, "
                         f"MSE: {mse_score:.4f}, PSNR: {psnr_score:.4f}")
        for full_image_id in sorted(patch_scores.keys()):
            scores = patch_scores[full_image_id]
            y_true.append(label)
            y_scores_ssim.append(max(scores['ssim']))
            y_scores_mse.append(max(scores['mse']))
            y_scores_psnr.append(max(scores['psnr']))

    # Process good images
    process_folder(good_original_folder, good_reconstructed_folder, 0)
    
    # Process bad images
    process_folder(bad_original_folder, bad_reconstructed_folder, 1)
    
    return np.array(y_true), np.array(y_scores_ssim), np.array(y_scores_mse), np.array(y_scores_psnr)

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    
    return roc_auc

def find_optimal_threshold_YoudenJ(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    return best_threshold

def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    distances = np.sqrt(fpr**2 + (1-tpr)**2)
    
    optimal_idx = np.argmin(distances)
    
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

def find_threshold_for_perfect_recall(y_true, y_scores):
    # Sort scores and corresponding true values
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    sorted_indices = np.argsort(y_scores)
    y_scores_sorted = y_scores[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    # Find the minimum score of all positive samples
    threshold = min(y_scores_sorted[y_true_sorted == 1])
    
    return threshold