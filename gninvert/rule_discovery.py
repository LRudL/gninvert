from datetime import datetime
import os
import torch as t
import torch_geometric as ptgeo
from functools import reduce

from gninvert.data_generation import get_TrainingData
from gninvert.gnns import GNN_full
from gninvert.hyperparamsearch import hpsearch
from gninvert.symbolic_regression import get_pysr_equations

def find_model(
        data, # expected format: gninvert.data_generation.TrainingData
        hp_save_location=False,
        model_save_location=False,
        nn_constructor=GNN_full,
        hyperparam_settings = None,
        hyperparam_overrides={},
        best_of = 1,
        seed = None,
        return_all = False
):
    if hyperparam_settings == None:
        hyperparam_settings = {
            # selected as important by decision tree on big hpsearch:
            'loss_func': t.nn.MSELoss(),
            'optimizer': 'adam',
            'regularization_coefficient': False,
            # other hyperparams:
            'starting_lr': 0.1,
            'lr_scheduler_dec_factor': 0.1,
            'lr_scheduler_patience': 25,
            'lr_scheduler_cooldown': 1,
            'batch_size': 2,
            'adam_weight_decay': 1e-6,

            # how patient are you?
            'epochs': 100,
            
            # ARGS TO gnn (in order):
            1: None,       # node_features
            2: None,       # message_featuers
            3: [16,16],       # (message_)hidden_sizes
            4: t.nn.GELU,  # (message_)nonlinearity
            5: True        # (message_)end_with_nonlinearity
        }
    if data.is_x_type_graph():
        node_features = data.train_ds()[0][0].x.shape[1]
        print(f"Number of node features: {node_features}")
        if hyperparam_settings[1] == None: # first arg to GNN is node feature number
            hyperparam_settings[1] = node_features
            if hyperparam_settings[2] == None: # second arg to GNN is message feature number
                print(f"Defaulting to {node_features} message features")
                hyperparam_settings[2] = node_features
                # ^ this is a good catch-all choice
                
    for param, value in hyperparam_overrides.items():
        if data.is_x_type_graph():
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
        hyperparam_settings, nn_constructor, training_data = data, verbose = True,
        rerun_times = best_of,
        seed = seed
    )

    if hp_save_location:
        t.save(hp_results, hp_save_location)
        print(f"Saved model results list to {hp_save_location}")

    if model_save_location:
        t.save(hp_results[0]['model'], model_save_location)
        print(f"Saved model with lowest validation loss to {model_save_location}")
    
    if return_all:
        return hp_results
    return hp_results[0]['model']

def find_rule_for_fn(
        fn,
        fn_arg_dims,
        arg_names = None,
        hyperparams = {},
        return_all = False
):
    variable_names = []
    for arg, dim in zip(arg_names, fn_arg_dims):
        if dim == 1:
            variable_names.append(arg)
        else:
            for i in range(1, dim + 1):
                variable_names.append(arg + str(i))
    pysr_model = get_pysr_equations(
        function = fn,
        dimensions = fn_arg_dims,
        variable_names = variable_names,
        constraints = None
    )
    if return_all:
        return pysr_model
    return pysr_model.get_best()

def find_rules_for_model(
        model,
        arg_dims = None,
        #save_location = False,
        return_all = False
):
    if hasattr(model, 'message') and hasattr(model, 'update') and hasattr(model, 'propagate'):
        # then this should really be a GN
        message_arg_dims = [model.node_features, model.node_features]
        update_arg_dims  = [model.node_features, model.message_features]
        message_rule = find_rule_for_fn(
            model.message,
            message_arg_dims,
            arg_names = ["xt", "xs"], # abbreviations for x_target and x_source
            return_all = True
        )
        update_rule = find_rule_for_fn(
            model.update,
            update_arg_dims,
            arg_names = ["xt", "a"], # abbreviations for x_target and aggregation
            return_all = True
        )
        to_return = (message_rule, update_rule)
    else:
        to_return = find_rule_for_fn(model, arg_dims, return_all = True)

    #if save_location != False:
    #    t.save(to_return, save_location)

    if return_all:
        return to_return
    if type(to_return) == tuple:
        return (to_return[0].get_best(), to_return[1].get_best())
    return to_return.get_best()

def discover_rules(
        data, # expected format: gninvert.data_generation.TrainingData
        save_to_file = True,
        file_location = "runs",
        run_name = None,
        nn_constructor = GNN_full,
        hyperparam_settings = None,
        hyperparam_overrides = {},
        models_per_hp_setting = 1
):
    if run_name == None:
        run_name = datetime.now().strftime("gninversion_%Y-%m-%d_%H:%M:%S")
    if save_to_file:
        if not os.path.isdir(file_location):
            os.mkdir(file_location)
    if save_to_file:
        os.makedirs(file_location + "/" + run_name)
        hpsearch_save_location = file_location + "/" + run_name + "/hpsearch"
        model_save_location = file_location + "/" + run_name + "/model"
        #sr_save_location = file_location + "/" + run_name + "/sr"
        #os.mkdir(sr_save_location)
    else:
        hpsearch_save_location = False
        model_save_location = False
        #sr_save_location = False

    (xs_are_graphs, ys_are_graphs) = data.are_types_graphs()
    
    print("TRAINING")
    
    model = find_model(
        data,
        hpsearch_save_location,
        model_save_location,
        nn_constructor = nn_constructor,
        hyperparam_settings = hyperparam_settings,
        hyperparam_overrides = hyperparam_overrides,
        best_of = models_per_hp_setting
    )

    print("INVERTING")

    rules = find_rules_for_model(
        model,
        arg_dims = None if xs_are_graphs and ys_are_graphs else data.train_ds()[0][0].shape[0]#,
        #save_location = sr_save_location
    )

    return rules

def invert_gn(
        gn,
        save_to_file=True,
        file_location="runs",
        run_name=None,
        nn_constructor=GNN_full,
        hyperparam_settings=None,
        hyperparam_overrides={},
        models_per_hp_setting=1
):
    data = get_TrainingData(gn, big=True)
    return discover_rules(
        data=data,
        save_to_file=save_to_file,
        file_location=file_location,
        run_name=run_name,
        nn_constructor=nn_constructor,
        hyperparam_settings=hyperparam_settings,
        hyperparam_overrides=hyperparam_overrides,
        models_per_hp_setting=models_per_hp_setting
    )
