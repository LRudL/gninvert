from datetime import datetime
import os
import torch as t
from functools import reduce

from gninvert.gnns import GNN_full
from gninvert.hyperparamsearch import hpsearch

def find_model(
        gn_data,
        hp_save_location=False,
        model_save_location=False,
        gnn=GNN_full,
        hyperparam_settings = {
            # selected as important by decision tree on big hpsearch:
            'loss_func': t.nn.MSELoss(),
            'optimizer': 'adam',
            'regularization_coefficient': False,

            # other hyperparams:
            'starting_lr': 0.1,
            'lr_scheduler_dec_factor': 0.2,
            'lr_scheduler_patience': 30,
            'lr_scheduler_cooldown': 30,
            'batch_size': 16,
            'adam_weight_decay': 1e-6,

            # how patient are you?
            'epochs': 100,
            
            # ARGS TO gnn (in order):
            1: None,       # node_features
            2: None,       # message_featuers
            3: [64],       # (message_)hidden_sizes
            4: t.nn.GELU,  # (message_)nonlinearity
            5: True        # (message_)end_with_nonlinearity
        },
        hyperparam_overrides={},
        best_of = 1,
        seed = None
):
    node_features = gn_data.train_ds()[0][0].x.shape[1]
    print(f"Number of node features: {node_features}")
    if hyperparam_settings[1] == None: # first arg to GNN is node feature number
        hyperparam_settings[1] = node_features
    if hyperparam_settings[2] == None: # second arg to GNN is message feature number
        print(f"Defaulting to {node_features} message features")
        hyperparam_settings[2] = node_features
        # ^ this is a good catch-all choice
    
    for param, value in hyperparam_overrides.items():
        if param == "node_features":
            param = 1
        if param == "message_features":
            param = 2
        hyperparam_settings[param] = value

    hp_settings_all_lists = reduce(
        lambda a, b : a and b,
        [
            type(hp_val) == list or type(hp_val) == tuple
            for hp_val in hyperparam_settings.values()
        ]
    )

    if hp_settings_all_lists:
        print("Every hyperparameter setting is a list of options. Running hyperparameter search ...")
    else:
        print("Only one hyperparameter setting found; will not run a hyperparameter search.")
        # ... but we will still interface with hpsearch, because that is convenient, so:
        hyperparam_settings = {param : [value] for (param, value) in hyperparam_settings.items()}

    hp_results = hpsearch(
        hyperparam_settings, gnn, training_data = gn_data, verbose = True,
        rerun_times = best_of,
        seed = seed
    )

    return hp_results

def discover_rules(
        gn_data,
        save_to_file = True,
        file_location = "../runs",
        run_name = None,
        gnn = GNN_full,
        hyperparam_settings = None,
        hyperparam_overrides = {}
):
    if run_name == None:
        run_name = datetime.now().strftime("gninversion_%Y-%m-%d_%H:%M:%S")
    if save_to_file:
        if not os.path.isdir(file_location):
            os.mkdir(file_location)
    if save_to_file:
        hpsearch_save_location = file_location + "/hpsearch"
        model_save_location = file_location + "/model"
        sr_save_location = file_location + "/sr"
    else:
        hpsearch_save_location = False
        model_save_location = False
        sr_save_location = False

    print("TRAINING GNN")
    
    model, hyperparams = find_model(
        gn_data,
        hpsearch_save_location,
        model_save_location,
        gnn = gnn,
        hyperparam_settings = hyperparam_settings,
        hyperparam_overrides = hyperparam_overrides
    )

    print("INVERTING GNN")

    

    
