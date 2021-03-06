import torch as t
from gninvert.gns import MultiDiffusionGN
from gninvert.gnns import LinearGNN
from gninvert.graph_compare import model_steps_compare, model_pred_acc_fn_maker, model_last_loss_fn
from gninvert.functions import generate_training_data, copy_from
from gninvert.training import fit
import gninvert.dtree as dtree
import gninvert.data_generation
import itertools
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

def param_settings(d):
    param_tuples = itertools.product(*d.values())
    param_dicts = [
        {
            param_name: ptuple[list(d.keys()).index(param_name)]
            for param_name in d.keys()
        }
        for ptuple in param_tuples
    ]
    return param_dicts 

def train_on_param_settings(settings, model, model_eval, training_data):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    if 'optimizer' not in settings.keys():
        settings['optimizer'] = 'adam'
    if settings['optimizer'] == 'adam':
        optim = t.optim.Adam(
            model.parameters(),
            lr=settings['starting_lr'],
            weight_decay=settings['adam_weight_decay'],
            betas=settings.get('adam_betas', (0.9, 0.99))
        )
    elif settings['optimizer'] == 'sgd':
        optim = t.optim.SGD(
            model.parameters(),
            lr=settings['starting_lr'],
            momentum=settings.get('momentum', 0),
            dampening=settings.get('dampening', 0)
        )

    if 'lr_scheduler_dec_factor' in settings.keys():
        if 'lr_scheduler_patience' not in settings.keys():
            raise Exception("LR scheduler patience not set.")
        if 'lr_scheduler_cooldown' not in settings.keys():
            settings['lr_scheduler_cooldown'] = 0
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            patience=settings['lr_scheduler_patience'],
            cooldown=settings['lr_scheduler_cooldown'],
            factor=settings['lr_scheduler_dec_factor']
        )

    perf_history = fit(
        epochs=settings['epochs'],
        model=model,
        loss_func=settings['loss_func'],
        opt=optim,
        train_ds=training_data.train_ds(),
        valid_ds=training_data.valid_ds(),
        batch_size=settings['batch_size'],
        lr_scheduler=scheduler,
        progress_bar=True,
        regularization=settings.get('regularization_coefficient', False),
        reg_norm=settings.get('regularization_norm', 1)
    )

    return model, model_eval(model) if model_eval != None else None, perf_history

def hpsearch(
        params,
        model_constructor,
        model_score_fn=None, training_data=None, gn=None,
        verbose=False,
        rerun_times=1,
        seed = None
):
    """
    `params` is a dictionary of parameters. See train_on_param_settings to see which
    values should be included. Parameters with integer keys are interpreted as arguments
    to `model_constructor`.
    
    `model_constructor` gets called with, in ascending order according to the (integer) key,
    all params that have an integer key in `params` to produce the model to train.
    
    (optional) `model_score_fn` gives a score (bigger is better) to trained models.
    Note that validation loss history is always returned.

    (optional) `training_data` is the data to use for training.

    (optional) `gn` is the graph network. If `training_data` is `None`,
    the graph network is used to generate training data using default settings.
    """
    if seed != None:
        t.manual_seed(seed)
    if gn != None and training_data == None:
        training_data = get_TrainingData(gn, graphs=20, big=True)
    if training_data == None:
        raise Exception("No training data on which to train in hpsearch!")
    settings_list = param_settings(params)
    results = []
    for settings in tqdm(settings_list):
        model_arg_dict = {k : settings[k] for k in settings.keys() if type(k) == int}
        model_args = [
            pair[1] for pair in
            sorted([(k, model_arg_dict[k]) for k in model_arg_dict.keys()],
                   key = lambda pair : pair[0])
        ]
        for _ in tqdm(range(rerun_times), leave=False):
            model = model_constructor(*model_args)
            final_model, eval_val, perf_history = train_on_param_settings(
                settings, model, model_score_fn, training_data
            )
            results.append({
                "settings": settings,
                "model": final_model,
                "score": eval_val,
                "val_loss_history": perf_history
            })
    if model_score_fn == None:
        sort_fn = lambda x : x["val_loss_history"][-1]
    else:
        sort_fn = lambda x : x["score"]
    results =  sorted(results, key = sort_fn)
    if verbose:
        view_hp_results_graph(results)
            
    return results

def view_hp_results_graph(results, ordered=True):
    val_series = [res['val_loss_history'] for res in results]
    if ordered:
        n = min(len(results), 3)
        for i in range(n, len(results)):
            plt.plot(val_series[i], alpha=0.33)
        for i in range(n):
            plt.plot(val_series[i], linewidth=3, c=['black', 'purple', 'blue'][i])
    else:
        for p in val_series:
            plt.plot(p)
    plt.yscale('log')
    plt.title('Validation loss histories in the hyperparameter search')
    plt.ylabel('Validation loss')
    plt.xlabel('Epoch')

def hp_argmin(fn):
    def argminer(hpres):
        if len(hpres) == 0:
            raise Exception("Empty hyperparameter results object!")
        minimum = float('inf')
        argmin = None
        for res in hpres:
            val = fn(res)
            if val < minimum:
                argmin = res
                minimum = val
        return (argmin, minimum)
    return argminer
    

def hp_stats(hpres, gn):
    val_minimising_res, val_min = hp_argmin(model_last_loss_fn)(hpres)
    err_minimising_res, err_min = hp_argmin(model_pred_acc_fn_maker(gn))(hpres)
    return {
        'final_val_loss': {
            'minimised_by': val_minimising_res['settings'],
            'minimum': val_min
        },
        'prediction_error': {
            'minimised_by': err_minimising_res['settings'],
            'minimum': err_min
        }
    }

def get_hyperparam_dtree(
        hp_results,
        val_func = lambda res : res['val_loss_history'][-1],
        eq_threshold = 1.0
):
    hp_results = list(filter(lambda res : not np.isnan(res['val_loss_history'][-1]), hp_results))
    params = {param_name : () for param_name in hp_results[0]['settings'].keys()}
    for res in hp_results:
        for param_name in res['settings'].keys():
            if res['settings'][param_name] not in params[param_name]:
                params[param_name] = tuple(list(params[param_name]) + [res['settings'][param_name]])
    params_with_choice = [param_name for param_name in params.keys()
                          if len(params[param_name]) > 1]
    attributes = {dtree.Attribute(param_name, tuple(params[param_name]))
                  for param_name in params_with_choice}
    examples = [dtree.Example(copy_from(res['settings'],
                                        params_with_choice),
                              val_func(res)) for res in hp_results]
    tree = dtree.decision_tree(
        examples, attributes, examples,
        dtree.log_variance_reduction_importance,
        dtree.log_based_representative_value,
        dtree.approx_equal_gen(threshold=eq_threshold)
    )
    return tree
    
