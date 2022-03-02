import torch as t
from gninvert.gns import MultiDiffusionGN
from gninvert.gnns import LinearGNN
from gninvert.graph_compare import model_steps_compare
from gninvert.functions import generate_training_data
from gninvert.training import fit
import gninvert.data_generation
import itertools
import tqdm.notebook as tq

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
    optim = t.optim.Adam(
        model.parameters(),
        lr=settings['starting_lr'],
        weight_decay=settings['adam_weight_decay']
    )

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
        progress_bar = True,
        regularization=settings.get('regularization_coefficient', False),
        reg_norm=settings.get('regularization_norm', 1)
    )

    return model, model_eval(model), perf_history

def hpsearch(params, model_constructor, model_score_fn, training_data=None):
    settings_list = param_settings(params)
    results = []
    for settings in tq.tqdm(settings_list):
        final_model, eval_val, perf_history = train_on_param_settings(
            settings, model_constructor(), model_score_fn, training_data
        )
        results.append({
            "settings": settings,
            "model": final_model,
            "score": eval_val,
            "val_loss_history": perf_history
        })
    return sorted(results, key = lambda x : x["score"])


