import torch as t

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

params = {
    'starting_lr': [5e-2],
    'lr_scheduler_dec_factor': [0.2],
    'lr_scheduler_patience': [20],
    'lr_scheduler_cooldown': [10],
    'batch_size': [10, 50],
    'adam_weight_decay': [5e-8],
    'epochs': [5],
    'loss_func': [t.nn.L1Loss()],
    'regularization_coefficient': [False],
    'regularization_norm': [1]
}

t.save(params, 'parameters')
