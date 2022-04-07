import torch as t
from tqdm import tqdm

def graphs_loss_func(model, xb, yb, node_loss_func = t.nn.MSELoss()):
    result = [model(gdata) for gdata in xb]
    losses = [node_loss_func(node_results, ygdata.x)
              for (node_results, ygdata) in
              zip(result, yb)]
    loss = sum(losses)
    return loss

def loss_batch(
        model,
        loss_func,
        xb, yb,
        opt=None,
        regularization=False,
        reg_norm=1
):
    if t.is_tensor(xb) and t.is_tensor(yb):
        loss = loss_func(model(xb), yb)
    elif not t.is_tensor(xb) and not t.is_tensor(yb): # ... then assume they're graph batches
        loss = graphs_loss_func(model, xb, yb, node_loss_func = loss_func)
    else:
        raise Exception("Trying to compare graph batch with non-graph batch in loss_batch")
    if regularization != False:
        reg = None
        for W in model.parameters():
            if reg is None:
                reg = W.norm(reg_norm)
            else:
                reg = reg + W.norm(reg_norm)
        loss += regularization * reg
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # loss.item(): pure numeric value of loss
    # also returns length of the batch, 
    return loss.item(), len(xb)

def fit(
        epochs, model, loss_func, opt, train_ds, valid_ds, batch_size=1,
        lr_scheduler = None,
        return_early_on_lr = None,
        progress_bar = False,
        return_lr = False,
        regularization=False,
        reg_norm=1
):
    train_x, train_y = train_ds
    valid_x, valid_y = valid_ds
    perf_history = []
    lr_history = []
    epoch_range = tqdm(range(epochs), leave=False) if progress_bar else range(epochs)

    if len(train_x) < batch_size:
        raise Exception(f"BATCH SIZE TOO LARGE! Train example length is only {len(train_x)}, batch size {batch_size}")
    
    for epoch in epoch_range:
        
        model.train()
        
        for b in range(len(train_x) // batch_size):
            i = b * batch_size
            end_i = (b+1) * batch_size
            xb = train_x[i : end_i]
            yb = train_y[i : end_i]
            loss_batch(
                model, loss_func,
                xb, yb,
                opt,
                regularization=regularization,
                reg_norm=reg_norm
            )

        
        model.eval()

        with t.no_grad():
            val_loss, nums = loss_batch(model,
                                        loss_func,
                                        valid_x,
                                        valid_y)
            perf_history.append(val_loss)
            if lr_scheduler != None:
                lr_scheduler.step(val_loss)
                lr = lr_scheduler.optimizer.param_groups[0]['lr']
                lr_history.append(lr)
                if return_early_on_lr is not None:
                    if lr < return_early_on_lr:
                        return perf_history
            if epoch % 10 == 0 or epoch == epochs - 1:
                if not progress_bar:
                    print(epoch, val_loss, lr)

    if return_lr:
        return perf_history, lr_history
    return perf_history
