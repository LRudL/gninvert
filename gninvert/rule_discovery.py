from datetime import datetime
import os
import torch as t
import torch_geometric as ptgeo
from functools import reduce
import dill

from gninvert.functions import sort_with
from gninvert.graph_compare import model_steps_compare, model_compare
from gninvert.data_generation import get_TrainingData
from gninvert.gnns import GNN_full
from gninvert.gns import RecoveredGN
from gninvert.hyperparamsearch import hpsearch, view_hp_results_graph
from gninvert.symbolic_regression import get_pysr_equations

HPSEARCH_SAVE_LOC = "/hpsearch"
MODEL_SAVE_LOC = "/model"
SR_SAVE_LOC = "/sr"
GN_SAVE_LOC = "/gn"

def make_dir_for_run(file_location, run_name):
    if not os.path.isdir(file_location):
        os.makedirs(file_location, exist_ok=True)
    if not os.path.isdir(file_location + "/" + run_name):
        os.makedirs(f"{file_location}/{run_name}", exist_ok=True)

def find_model(
        data, # expected format: gninvert.data_generation.TrainingData
        hp_save_location=False,
        model_save_location=False,
        nn_constructor=GNN_full,
        hyperparam_settings = None,
        hyperparam_overrides={},
        best_of = 1,
        seed = None,
        return_all = False,
        model_compare_fn = lambda hp_result_obj : hp_result_obj['val_loss_history'][-1]
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
            'epochs': 200,
            
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
        if type(hyperparam_settings[1]) == list or type(hyperparam_settings[1]) == tuple:
            hyperparam_settings[1] = [node_features if n == None else n
                                      for n in hyperparam_settings[1]]
    
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

    hp_results = sort_with(model_compare_fn, hp_results)

    if hp_save_location:
        t.save(hp_results, hp_save_location)
        print(f"Saved model results list to {hp_save_location}")

    if model_save_location:
        t.save(hp_results[0]['model'], model_save_location)
        print(f"Saved model with lowest validation loss to {model_save_location}")
    
    if return_all:
        return hp_results
    return hp_results[0]

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
        save_location = False,
        data_trained_on = None
):
    if hasattr(model, 'message') and hasattr(model, 'update') and hasattr(model, 'propagate'):
        # then this should really be a GN
        message_arg_dims = [model.node_features, model.node_features]
        update_arg_dims  = [model.node_features, model.message_features]
        message_rule = find_rule_for_fn(
            model.message,
            message_arg_dims,
            arg_names = ["xt", "xs"] # abbreviations for x_target and x_source
        )
        update_rule = find_rule_for_fn(
            model.update,
            update_arg_dims,
            arg_names = ["xt", "a"] # abbreviations for x_target and aggregation
        )
        to_return = RecoveredGN(message_rule, update_rule, data_trained_on)
        # cannot save directly in any normal sensible way because pickling doesn't work on
        # PySR's lambda_format equations
        # Therefore, RecoveredGN implements a .save and a (static!) .load method
        # for saving and loading in a way that works around these issues
        # (each taking string path as an arg)
        to_return.save(save_location)
    else:
        to_return = find_rule_for_fn(model, arg_dims, return_all = False)
        t.save(to_return, save_location)

    return to_return



def discover_rules(
        data, # expected format: gninvert.data_generation.TrainingData
        save_to_file = True,
        file_location = "runs",
        run_name = None,
        nn_constructor = GNN_full,
        hyperparam_settings = None,
        hyperparam_overrides = {},
        models_per_hp_setting = 1,
        model_compare_fn = lambda hp_results : hp_results[0]['model'],
        skip_invert = False
):
    if run_name == None:
        run_name = datetime.now().strftime("gninversion_%Y-%m-%d_%H:%M:%S")
    if save_to_file:
        make_dir_for_run(file_location, run_name)
        hpsearch_save_location = file_location + "/" + run_name + HPSEARCH_SAVE_LOC
        model_save_location = file_location + "/" + run_name + MODEL_SAVE_LOC
        sr_save_location = file_location + "/" + run_name + SR_SAVE_LOC
    else:
        hpsearch_save_location = False
        model_save_location = False
        sr_save_location = False

    (xs_are_graphs, ys_are_graphs) = data.are_types_graphs()
    
    print("TRAINING")
    
    model_res = find_model(
        data,
        hpsearch_save_location,
        model_save_location,
        nn_constructor = nn_constructor,
        hyperparam_settings = hyperparam_settings,
        hyperparam_overrides = hyperparam_overrides,
        best_of = models_per_hp_setting,
        model_compare_fn = model_compare_fn
    )

    if skip_invert:
        print("skip_invert set to True; EXITING EARLY.")
        return model_res
    
    print("INVERTING")

    arg_dims = None if xs_are_graphs and ys_are_graphs else [data.train_ds()[0][0].shape[0]]
    rules = find_rules_for_model(
        model_res['model'],
        arg_dims = arg_dims,
        save_location = sr_save_location,
        data_trained_on = data
    )

    if save_to_file:
        print(f"Everything saved at {file_location}/{run_name}")

    return rules
def invert_gn(
        gn,
        save_to_file=True,
        file_location="runs",
        run_name=None,
        nn_constructor=GNN_full,
        hyperparam_settings=None,
        hyperparam_overrides={},
        models_per_hp_setting=1,
        graphs_in_training_data=20,
        training_graph_size=1000,
        model_criterion = 'simulation',
        skip_invert = False
):
    if save_to_file == True:
        make_dir_for_run(file_location, run_name)
        t.save(gn, file_location + "/" + run_name + GN_SAVE_LOC)
    data = get_TrainingData(gn,
                            graphs=graphs_in_training_data,
                            graph_size=training_graph_size,
                            big=True)
    if model_criterion == 'simulation':
        model_compare_fn = lambda res : model_steps_compare(res['model'],
                                                            gn,
                                                            gdata=data.a_graph(),
                                                            iterations=6)['absolute']['avg_difs'][-1]
    elif model_criterion == 'loss':
        model_compare_fn = lambda res : res['val_loss'][-1]
    else:
        raise Exception(f"Undefined model_criterion {model_criterion} in invert_gn. 'simulation' and 'loss' are valid values.")
    return discover_rules(
        data=data,
        save_to_file=save_to_file,
        file_location=file_location,
        run_name=run_name,
        nn_constructor=nn_constructor,
        hyperparam_settings=hyperparam_settings,
        hyperparam_overrides=hyperparam_overrides,
        models_per_hp_setting=models_per_hp_setting,
        model_compare_fn=model_compare_fn,
        skip_invert=skip_invert
    )

def view_run_results(fpath):
    hpresults = t.load(fpath + HPSEARCH_SAVE_LOC)
    model = t.load(fpath + MODEL_SAVE_LOC)
    try:
        sr = t.load(fpath + SR_SAVE_LOC)
    except:
        sr = "couldn't load SR"
    gn = t.load(fpath + GN_SAVE_LOC)
    view_hp_results_graph(hpresults, ordered=True)
    model_compare(gn, model)
    #print(model)
    print(sr)
    #return hpresults, model, sr
