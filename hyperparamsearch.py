import torch as t
from gns import MultiDiffusionGN
from gnns import LinearGNN
from graph_compare import model_steps_compare
from functions import generate_training_data
from training import fit
import itertools
import tqdm.notebook as tq


diffusionGN = MultiDiffusionGN([0.1, 0.1])

x_train, y_train = generate_training_data(diffusionGN)

train_fraction = 0.75

train_i = round(len(x_train) * train_fraction)
train_x = x_train[0:train_i]
valid_x = x_train[train_i:]
train_y = y_train[0:train_i]
valid_y = y_train[train_i:]


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

def train_on_param_settings(settings, model, model_eval):
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
        train_ds=(train_x, train_y),
        valid_ds= (valid_x, valid_y),
        batch_size=settings['batch_size'],
        lr_scheduler=scheduler,
        progress_bar = True
    )

    return model, model_eval(model), perf_history

def hpsearch(params, model_constructor, model_score_fn):
    settings_list = param_settings(params)
    results = []
    for settings in tq.tqdm(settings_list):
        final_model, eval_val, perf_history = train_on_param_settings(
            settings, model_constructor(), model_score_fn
        )
        results.append({
            "settings": settings,
            "model": final_model,
            "score": eval_val,
            "val_loss_history": perf_history
        })
    return sorted(results, key = lambda x : x["score"])
