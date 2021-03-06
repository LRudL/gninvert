import argparse
import os 

import torch as t

from gninvert.rule_discovery import find_model, find_rules_for_model, invert_gn
from gninvert.gns import SingleDiffusionGN, MultiDiffusionGN, ActivatorInhibitorGN, FullActInhGN
from gninvert.gnns import GNN_full

os.makedirs("models", exist_ok=True)
os.makedirs("runs", exist_ok=True)

all_args = argparse.ArgumentParser()

all_args.add_argument("-n", "--name", required=True,
                      help="name prefix under which to save runs")
all_args.add_argument("-m", "--models", required=False,
                      help="which models to invert")
all_args.add_argument("-p", "--hps", required=False,
                      help="which hp sets to use (same order as models)")
all_args.add_argument("-s", "--skip", required=False,
                      help="only do training, no SR")
all_args.add_argument("-S", "--skiptrain", required=False,
                      help="only do SR, no training")
all_args.add_argument("-M", "--modelpath", required=False,
                      help="replace model with model from this path")
all_args.add_argument("-c", "--criterion", required=False,
                      help="criterion for selecting best model, either 'simulation' or 'loss'",
                      default='simulation')
all_args.add_argument("-r", "--runs", required=False,
                      help="how many runs per HP setting",
                      default=4)

args = vars(all_args.parse_args())

models = { # remember to add to model_message_features too!
    'diff1' : SingleDiffusionGN(diffusion_constant=0.1),
    'diff2' : MultiDiffusionGN(diffusion_constants=[0.1, 0.1]),
    'diff3' : MultiDiffusionGN(diffusion_constants=[0.1, 0.1, 0.1]),
    'vardiff3': MultiDiffusionGN(diffusion_constants=[0.15, 0.1, 0.05]), 
    'act_inh_simple': ActivatorInhibitorGN(act_diff_const=0.1,
                                           inh_diff_const=0.05,
                                           growth_const=0.05,
                                           eps=0.0),
    'act_inh_simple_with_eps': ActivatorInhibitorGN(act_diff_const=0.1,
                                             inh_diff_const=0.05,
                                             growth_const=0.05,
                                             eps=0.1),
    'act_inh_full': FullActInhGN(spatial_const=10,
                                 temporal_const=0.01,
                                 growth_alpha=10,
                                 growth_rho=1,
                                 growth_scale=0.05,
                                 reaction_const=0.2,
                                 reference_conc=2)
}

model_message_features = {
    'diff1': 1,
    'diff2': 2,
    'diff3': 3,
    'vardiff3': 3,
    'act_inh_simple': 2,
    'act_inh_simple_with_eps': 2,
    'act_inh_full': 2
}

hps = {
    'full': {
        'loss_func': [t.nn.MSELoss(), t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-5, 1e-3],
        'regularization_norm': [1, 2],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.2],
        'lr_scheduler_patience': [25, 75],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [1e-7],
        'epochs': [200],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[64], [256], [1024], [32, 32], [16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [True, False] # nonlinearity at end
    },
    'general': {
        'loss_func': [t.nn.MSELoss(), t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-5, 1e-3],
        'regularization_norm': [1, 2],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [100],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [500],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[16], [64], [256], [16, 16], [64, 64], [256, 256], [16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'minimal': {
        'loss_func': [t.nn.MSELoss()],
        'optimizer': ['adam'],
        'regularization_coefficient': [False],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.2],
        'lr_scheduler_patience': [25],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [1e-7],
        'epochs': [200],
        1: [None],
        2: [None],
        3: [[64]],
        4: [t.nn.GELU],
        5: [True]
    },
    'linear': {
        'loss_func': [t.nn.MSELoss()],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-6, 1e-3],
        'regularization_norm': [1, 2],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [50],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [500],
        1: [None],
        2: [None],
        3: [[]],
        4: [t.nn.GELU],
        5: [False]
    },
    'paper': {
        'loss_func': [t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [1e-8],
        'regularization_norm': [2],
        'starting_lr': [0.1],                # improv
        'lr_scheduler_dec_factor': [0.1],    # improv
        'lr_scheduler_patience': [100],      # improv
        'lr_scheduler_cooldown': [1],        # improv
        'batch_size': [4],                   # improv
        'adam_weight_decay': [0],            # improv
        'adam_betas': [(0.7, 0.97), (0.8, 0.98), (0.85, 0.985), (0.9, 0.999), (0.97, 0.9997)], #improv
        'epochs': [500],                     # improv
        1: [None],
        2: [None],
        3: [[300, 300]],
        4: [t.nn.ReLU],
        5: [False]                           # improv
    },
    'diff1_precise': { # runs/PRECISE1_diff1 (BEST for diff1)
        'loss_func': [t.nn.MSELoss()],
        'optimizer': ['adam'],
        'regularization_coefficient': [1e-5],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [100],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [1500],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[64], [16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'diff2_search':  { # see runs/diff2_try2_diff2 for results
        'loss_func': [t.nn.MSELoss()],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-5, 1e-3],
        'regularization_norm': [1, 2],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.2],
        'lr_scheduler_patience': [75],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [1e-7],
        'epochs': [500],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[16], [64], [8, 8], [32, 32], [8, 8, 8], [16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'diff2_precise': { # diff2_try2_diff2
        'loss_func': [t.nn.MSELoss()],
        'optimizer': ['adam'],
        'regularization_coefficient': [False],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.2],
        'lr_scheduler_patience': [75],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [1e-7],
        'epochs': [1000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[16], [64], [8, 8, 8], [16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'diff2_tuned': { # PRECISE1_diff2 (BEST for diff2)
        'loss_func': [t.nn.MSELoss()],
        'optimizer': ['adam'],
        'regularization_coefficient': [False],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.2],
        'lr_scheduler_patience': [75],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [1e-7],
        'epochs': [2000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[16], [16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'act_inh_simple_search': { # searches_act_inh_simple
        'loss_func': [t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-3],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.2],
        'lr_scheduler_patience': [25, 75],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [1e-7],
        'epochs': [500],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[64], [256], [512], [64, 64], [256, 256], [32, 32, 32]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'act_inh_simple_search2': { # searches2_act_inh_simple
        'loss_func': [t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-3],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.2],
        'lr_scheduler_patience': [100],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [1e-7],
        'epochs': [1000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[32], [64], [128], [256], [512], [1024], [2048], [4096],
            [32,32], [64, 64], [128, 128],
            [16, 16, 16], [32, 32, 32], [64, 64, 64],
            [16, 16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'act_inh_simple_search3': { # runs/SEARCHES3_act_inh_simple
        'loss_func': [t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [200],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [2000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[32, 32, 32]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'act_inh_simple_shallow_search': { # runs/SHALLOW_act_inh_simple (interesting results!)
        'loss_func': [t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [200],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [2000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[32], [64], [128], [256], [512], [1024], [2048], [4096]],
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'act_inh_simple_bignet_search':  { # runs/searches_bignet_act_inh_simple
        'loss_func': [t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-3],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [200],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [1000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[256, 256],
            [16, 16, 16], [32, 32, 32],
            [16, 16, 16, 16]],
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'act_inh_full_search':  {
        'loss_func': [t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-5, 1e-3],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [200],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [2000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[256, 256], [32, 32, 32]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
    'act_inh_simple_eps_search1': {
        'loss_func': [t.nn.MSELoss(), t.nn.L1Loss(reduction="mean")],
        'optimizer': ['adam'],
        'regularization_coefficient': [False, 1e-3],
        'regularization_norm': [1],
        'starting_lr': [0.1],
        'lr_scheduler_dec_factor': [0.1],
        'lr_scheduler_patience': [100],
        'lr_scheduler_cooldown': [1],
        'batch_size': [2],
        'adam_weight_decay': [0],
        'epochs': [2000],
        1: [None], # node features - gets autofilled if None
        2: [None], # message features - gets autofilled if None
        3: [[16], [64], [256], [16, 16], [64, 64], [16, 16, 16]], # hidden sizes
        4: [t.nn.GELU], # nonlinearity
        5: [False] # nonlinearity at end
    },
}

skip_sr = args['skip'] != None 
if skip_sr:
    print("Warning: skipping symbolic regression step. No equations will be output.")
    print("To perform symbolic regression, do not pass the -s/--skip option")

skip_training = args['skiptrain'] != None
if skip_training:
    print("Warning: skipping training step and jumping straight to symbolic regression.")
    print("To perform training, do not pass the -S/--skiptrain option")

if args['modelpath'] != None:
    print(f"Loading custom model to train from {args['modelpath']} ...")
    t.load(args['modelpath'])
    print("Model loads!")

for i in range(len(args['models'].split(" "))):
    mname = args['models'].split(" ")[i]
    if mname not in models.keys():
        print(f"Skipping unknown model '{mname}'")
    else:
        print(f"\n\n\nWorking on model: {mname}")
        hp_settings = list(hps.values())[0]
        if 'hps' in args.keys() and args['hps'] != None:
            hpss = args['hps'].split(" ")
            if i < len(hpss):
                hp_settings = hps[hpss[i]]
        hp_settings[2] = [model_message_features[mname]]
        model = models[mname] if args['modelpath'] == None else t.load(args['modelpath'])            
        invert_gn(model,
                  save_to_file=True,
                  file_location="runs",
                  run_name=f"{args['name']}_{mname}",
                  nn_constructor=GNN_full,
                  hyperparam_settings=hp_settings,
                  models_per_hp_setting=int(args['runs']),
                  graphs_in_training_data=4,
                  training_graph_size=1024,
                  model_criterion=args['criterion'],
                  skip_invert=skip_sr,
                  skip_training=skip_training)
