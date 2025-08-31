import torch.utils.data as data
import numpy as np
import torch
import glob
import os
import imageio.v2 as imageio
import random
from torch.utils.data import DataLoader
from skimage import color as sc

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args[0]], _np2Tensor(args[1])

def get_patch(*args, patch_size=96, scale=[2], multi_scale=False):
    th, tw = args[-1].shape[:2] # target images size

    tp = patch_size  # patch size of target hr image
    ip = [patch_size // s for s in scale] # patch size of lr images

    # tx and ty are the top  and left coordinate of the patch 
    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)
    tx, ty = tx- tx % scale[0], ty - ty % scale[0]
    ix, iy = [ tx // s for s in scale], [ty // s for s in scale]
   
    lr = [args[0][i][iy[i]:iy[i] + ip[i], ix[i]:ix[i] + ip[i], :] for i in range(len(scale))]
    hr = args[-1][ty:ty + tp, tx:tx + tp, :]

    return [lr, hr]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args[0]], _augment(args[-1])

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args[0]], _set_channel(args[-1])

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale.copy()
        self.scale.reverse()
        
        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)

        lr, hr = set_channel(lr, hr, n_channels=self.args.n_colors)
        lr, hr = self.get_patch(lr, hr)
        
        lr_tensor, hr_tensor = np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )

        return lr_tensor, hr_tensor, filename
    
    def __len__(self):
        return self.dataset_length

    def _get_imgs_path(self, args):
        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr

    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            repeat = self.dataset_length // len(self.images_hr)
            self.random_border = len(self.images_hr) * repeat
        else:
            self.dataset_length = len(self.images_hr)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                # Support DIV2K-style LR_bicubic/X{s}/{filename}x{s}.png
                cand1 = os.path.join(self.dir_lr, 'LR_bicubic', f'X{s}', f'{filename}x{s}{self.ext[1]}')
                # Support simplified LR_{s}/{filename}.png
                cand2 = os.path.join(self.apath, f'LR_{s}', f'{filename}{self.ext[1]}')
                if os.path.exists(cand1):
                    path_lr = cand1
                elif os.path.exists(cand2):
                    path_lr = cand2
                else:
                    # Optional fallback: LR/{filename}.png
                    cand3 = os.path.join(self.apath, 'LR', f'{filename}{self.ext[1]}')
                    if os.path.exists(cand3):
                        path_lr = cand3
                    else:
                        raise FileNotFoundError(f"LR image not found for {filename} at scale {s}: tried {cand1}, {cand2}, {cand3}")
                names_lr[si].append(path_lr)

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        # self.apath = os.path.join(data_dir, self.name)
        self.apath = data_dir
        self.dir_hr = os.path.join(self.apath, 'HR')
        # Leave LR root at dataset path; actual LR path chosen in _scan
        self.dir_lr = self.apath
        self.ext = ('.png', '.png')

    def _get_index(self, idx):
        if self.train:
            if idx < self.random_border:
                return idx % len(self.images_hr)
            else:
                return np.random.randint(len(self.images_hr))
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.scale))]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = [imageio.imread(f_lr[idx_scale]) for idx_scale in range(len(self.scale))]
        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = augment(lr, hr)
        else:
            if isinstance(lr, list):
                ih, iw = lr[0].shape[:2]
            else:
                ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale[0], 0:iw * scale[0]]
            
        return lr, hr

class MVTec(SRData):
    def __init__(self, args, name='MVTec', train=True, benchmark=False):
        super(MVTec, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        # Use parent implementation with flexible LR handling
        super(MVTec, self)._set_filesystem(data_dir)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            # module_train = import_module('data.' + args.data_train.lower())
            # trainset = getattr(module_train, args.data_train)(args)
            trainset = MVTec(args, train=True)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )

        
        testset = MVTec(args, train=False)

        self.loader_test = DataLoader(
            testset,
            batch_size=args.batch_size,
            num_workers=args.n_threads,
            shuffle=False,
            pin_memory=not args.cpu
        )