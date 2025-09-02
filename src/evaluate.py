import argparse
import copy
import os
import sys
import yaml  # type: ignore[import-untyped]
import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
from pathlib import Path
from PIL import Image

from src.main import DRN, DRCT, setup_opt_drn, setup_opt_drct
from src.data import Data
from src.model import Model
from src.loss import Loss
from src.checkpoint import Checkpoint
from src.helpers import calculate_ssim, calculate_psnr


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(description='Evaluation entrypoint', parents=[pre])
    p.add_argument('--model-type', type=str, default='drct', choices=['drct', 'drn-l'])
    p.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec'])
    p.add_argument('--classe', type=str, default='grid')
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--data-root', type=str, default='data/mvtec_128')
    p.add_argument('--run-dir', type=str, default='')
    p.add_argument('--checkpoint', type=str, default='')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--output-dir', type=str, default='')
    p.add_argument('--save-images', action='store_true', default=True)
    p.add_argument('--workers', type=int, default=0 if sys.platform == 'darwin' else 4)

    if pre_args.config and os.path.isfile(pre_args.config):
        with open(pre_args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        p.set_defaults(**{k.replace('-', '_'): v for k, v in cfg.items()})

    return p.parse_args(argv)


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    if args.run_dir:
        cand = os.path.join(args.run_dir, 'model', 'model_best.pt')
        if os.path.isfile(cand):
            return cand
        cand = os.path.join(args.run_dir, 'model', 'model_latest.pt')
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError('Please provide --checkpoint or a valid --run-dir containing model/*.pt')


def evaluate_on_test(opt, checkpoint_model_path, output_dir: str, save_images: bool):
    # Build test loaders for good and bad
    def build_loader(split):
        eopt = copy.deepcopy(opt)
        eopt.test_only = True
        eopt.no_augment = True
        eopt.batch_size = 1
        eopt.data_dir = f'{opt.data_root}/{opt.classe}/test/{split}'
        eopt.data_test = f'mvtec_test_{split}'
        return Data(eopt).loader_test

    loader_good = build_loader('good')
    loader_bad = build_loader('bad')

    # Minimal environment to run the model forward
    ckp = Checkpoint(opt)
    model = Model(opt, ckp, dual_model=(opt.model_name == 'drn-l'))
    # Overwrite pre_train live (Model() already loaded based on opt.pre_train)
    # If needed, reload explicitly from checkpoint path
    model.load(pre_train=checkpoint_model_path, pre_train_dual='.', cpu=opt.cpu)

    class Runner:
        # Shim to reuse prepare from Trainer without importing full Trainer here
        def __init__(self, opt, model):
            self.opt = opt
            self.model = model
        def prepare(self, *args):
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                import torch
                if (not self.opt.cpu) and torch.cuda.is_available():
                    device = torch.device('cuda')
                elif (not self.opt.cpu) and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = torch.device('mps')
                else:
                    device = torch.device('cpu')
            if len(args) > 1:
                return [a.to(device) for a in args[0]], args[-1].to(device)
            return [a.to(device) for a in args[0]],

    import torch
    torch.no_grad().__enter__()
    t = Runner(opt, model)

    y_true = []
    sr_np = []
    hr_np = []
    filenames = []
    splits = []

    # Create output directories if saving
    if save_images:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def save_sr_image(sr_tensor, name: str, split: str, scale_value: int):
        # Convert tensor to uint8 image and save
        sr_u8 = sr_tensor[0].detach().mul(255 / opt.rgb_range).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_dir = Path(output_dir) / split / f"x{scale_value}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if sr_u8.shape[2] == 1:
            img = Image.fromarray(sr_u8[:, :, 0], mode='L')
        else:
            img = Image.fromarray(sr_u8, mode='RGB')
        img.save(str(out_dir / f"{name}.png"))

    def collect_pairs(dloader, label, split_name: str):
        for _, (lr, hr, fname) in enumerate(dloader):
            if hr.nelement() == 1:
                continue
            lr_t, hr_t = t.prepare(lr, hr)
            sr = t.model(lr_t[0])
            if isinstance(sr, list):
                sr = sr[-1]
            h, w = hr_t.shape[-2:]
            sr = sr[..., :h, :w]
            sr_u8 = sr[0].detach().mul(255 / opt.rgb_range).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            hr_u8 = hr_t[0].detach().mul(255 / opt.rgb_range).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            y_true.append(label)
            sr_np.append(sr_u8)
            hr_np.append(hr_u8)
            name = fname[0] if isinstance(fname, (list, tuple)) else str(fname)
            filenames.append(name)
            splits.append(split_name)
            if save_images:
                scale_value = opt.scale[-1] if isinstance(opt.scale, list) else int(opt.scale)
                save_sr_image(sr, name, split_name, scale_value)

    collect_pairs(loader_good, 0, 'good')
    collect_pairs(loader_bad, 1, 'bad')

    if len(set(y_true)) < 2:
        print('Test set lacks both classes; AUC not available')
        return

    # Determine SSIM window sizes
    min_dim = min(min(img.shape[0], img.shape[1]) for img in hr_np)
    max_w = max(3, min_dim - 3)
    window_sizes = [w for w in range(3, max_w + 1, 10) if w % 2 == 1] or [3]

    best_ws = window_sizes[0]
    best_auc = -1.0
    for ws in window_sizes:
        scores = []
        for sr_img, hr_img in zip(sr_np, hr_np):
            ssim_score = calculate_ssim(hr_img.astype(np.float32) / 255.0, sr_img.astype(np.float32) / 255.0, ws)
            scores.append(1 - ssim_score)
        auc_ssim = roc_auc_score(y_true, scores)
        if auc_ssim > best_auc:
            best_auc = auc_ssim
            best_ws = ws

    # Final metrics
    y_scores_ssim = []
    y_scores_mse = []
    y_scores_psnr = []
    for sr_img, hr_img in zip(sr_np, hr_np):
        sr_f = sr_img.astype(np.float32) / 255.0
        hr_f = hr_img.astype(np.float32) / 255.0
        ssim_score = calculate_ssim(hr_f, sr_f, best_ws)
        y_scores_ssim.append(1 - ssim_score)
        diff = (sr_f - hr_f)
        y_scores_mse.append(float(np.mean(diff * diff)))
        y_scores_psnr.append(calculate_psnr(hr_f, sr_f))

    auc_ssim = roc_auc_score(y_true, y_scores_ssim)
    auc_mse = roc_auc_score(y_true, y_scores_mse)
    auc_psnr = roc_auc_score(y_true, [-p for p in y_scores_psnr])

    print(f"Test AUCs - SSIM(best ws={best_ws}): {auc_ssim:.4f}, MSE: {auc_mse:.4f}, PSNR: {auc_psnr:.4f}")


def main(argv=None):
    args = parse_args(argv)

    # Build base options
    model_type = args.model_type
    ds = args.dataset
    class_name = args.classe
    img_resolution = args.resolution
    scale = args.scale

    # Channel count
    n_colors = 3 if (ds == 'mvtec' and class_name == 'carpet') else 1

    # Common fields
    best_auc = 0.0
    ssim_window_size = 11
    slurm = False
    epochs = 1
    batch_size = args.batch_size
    no_augment = True
    patch_size = img_resolution
    img_size = img_resolution // scale
    data_dir = f"{args.data_root}/{class_name}/train/good"  # not used directly; Data() will override per split
    save = './workspace/eval'  # lightweight temp dir for Checkpoint utilities
    data_range = ''
    test_every = 1
    print_every = 1
    patience = 1
    min_delta = 0.0
    n_threads = args.workers
    loss = '1*L1'

    ckpt_path = resolve_checkpoint(args)

    if model_type == 'drn-l':
        opt = DRN()
        opt = setup_opt_drn(opt, best_auc, ssim_window_size, ds, class_name, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, ckpt_path, '.', loss)
    else:
        opt = DRCT()
        opt = setup_opt_drct(opt, best_auc, ssim_window_size, ds, class_name, slurm, scale, no_augment, n_colors, epochs, batch_size, patch_size, img_size, data_dir, save, data_range, test_every, print_every, patience, min_delta, n_threads, ckpt_path, loss)

    # Device override
    if args.device == 'cpu':
        opt.cpu = True

    # Add roots for evaluation convenience
    opt.model_name = model_type
    opt.data_root = args.data_root

    # Determine output directory
    if args.output_dir:
        out_dir = args.output_dir
    elif args.run_dir:
        out_dir = os.path.join(args.run_dir, 'eval_results')
    else:
        out_dir = './workspace/eval_results'

    evaluate_on_test(opt, ckpt_path, out_dir, args.save_images)


if __name__ == "__main__":
    main()


