import torch as t
from gninvert.gns import MultiDiffusionGN
from gninvert.gnns import LinearGNN
from gninvert.graph_compare import model_steps_compare
from gninvert.functions import generate_training_data
from gninvert.training import fit
import gninvert.data_generation
import itertools
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

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
            weight_decay=settings['adam_weight_decay']
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
        verbose=False
):
    """
    `params` is a dictionary of parameters. See train_on_param_settings to see which
    values should be included.
    
    `model_constructor` is either:
    1. a function that generates a new copy of the model to train (w/o any args)
    2. a list of such functions
    
    (optional) `model_score_fn` gives a score (bigger is better) to trained models.
    Note that validation loss history is always returned.

    (optional) `training_data` is the data to use for training.

    (optional) `gn` is the graph network. If `training_data` is `None`,
    the graph network is used to generate training data using default settings.
    """
    if gn != None and training_data == None:
        training_data = get_TrainingData(gn)
    if training_data == None:
        raise Exception("No training data on which to train in hpsearch!")
    settings_list = param_settings(params)
    results = []
    for settings in tqdm(settings_list):
        model_list = [model_constructor] if callable(model_constructor) \
            else model_constructor
        if type(model_list) is not list:
            raise Exception("model_constructor not callable and not a list.")
        for model_fn in tqdm(model_list):
            model = model_fn()
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
        val_series = [res['val_loss_history'] for res in results]
        for p in val_series:
            plt.plot(p)
        plt.yscale('log')
        plt.title('Validation loss histories in the hyperparameter search')
        plt.ylabel('Validation loss')
        plt.xlabel('Epoch')
            
    return results


