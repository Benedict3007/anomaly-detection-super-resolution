import torch.nn as nn
import torch
import os
import numpy as np
from drct import DRCT
from drn import DRN

class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval

        if nFeat is None:
            nFeat = opt.n_feats
        
        if in_channels is None:
            in_channels = opt.n_colors
        
        if out_channels is None:
            out_channels = opt.n_colors

        
        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x

def make_model(opt):
    if opt.model_name == 'drct':
        return DRCT(opt)
    elif opt.model_name == 'drn-l':
        return DRN(opt)
    else:
        print(f"No model with this name: {opt.model_name}")

class Model(nn.Module):
    def __init__(self, opt, ckp, dual_model=False):
        super(Model, self).__init__()
        print('Making model...')
        self.opt = opt
        self.scale = opt.scale
        self.idx_scale = 0
        self.self_ensemble = opt.self_ensemble
        self.cpu = opt.cpu
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.n_GPUs = opt.n_GPUs
        self.dual_model = dual_model

        self.model = make_model(opt).to(self.device)
        
        if self.dual_model:
            self.dual_models = []
            for _ in self.opt.scale:
                dual_model = DownBlock(opt, 2).to(self.device)
                self.dual_models.append(dual_model)
        
        self.load(opt.pre_train, opt.pre_train_dual, cpu=opt.cpu)

        if not opt.test_only:
            print(self.model, file=ckp.log_file)
            if self.dual_model:
                print(self.dual_models, file=ckp.log_file)
        
        # compute parameter
        num_parameter = self.count_parameters(self.model)
        ckp.write_log(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M")

    def forward(self, x, idx_scale=0):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module
    
    def get_dual_model(self, idx):
        if self.n_GPUs == 1:
            return self.dual_models[idx]
        else:
            return self.dual_models[idx].module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    
    def count_parameters(self, model):
        if self.opt.n_GPUs > 1:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self, path, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(path, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', 'model_best.pt')
            )
        if self.dual_model:
            #### save dual models ####
            dual_models = []
            for i in range(len(self.dual_models)):
                dual_models.append(self.get_dual_model(i).state_dict())
            torch.save(
                dual_models,
                os.path.join(path, 'model', 'dual_model_latest.pt')
            )
            if is_best:
                torch.save(
                    dual_models,
                    os.path.join(path, 'model', 'dual_model_best.pt')
                )

    def load(self, pre_train='.', pre_train_dual='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        #### load primal model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, weights_only=True, **kwargs),
                strict=False
            )

        if self.dual_model:
            #### load dual model ####
            if pre_train_dual != '.':
                print('Loading dual model from {}'.format(pre_train_dual))
                dual_models = torch.load(pre_train_dual, weights_only=True, **kwargs)
                for i in range(len(self.dual_models)):
                    self.get_dual_model(i).load_state_dict(
                        dual_models[i], strict=False
                    )