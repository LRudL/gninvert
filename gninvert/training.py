import torch as t
import tqdm.notebook as tq

def graphs_loss_func(model, xb, yb, node_loss_func = t.nn.MSELoss()):
    result = [model(gdata) for gdata in xb]
    losses = [node_loss_func(node_results, ygdata.x)
              for (node_results, ygdata) in
              zip(result, yb)]
    return sum(losses)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = graphs_loss_func(model, xb, yb, node_loss_func = loss_func)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # loss.item(): pure numeric value of loss
    # also returns length of the batch, 
    return loss.item(), len(xb)

def fit(
        epochs, model, loss_func, opt, train_ds, valid_ds, batch_size=10,
        lr_scheduler = None,
        return_early_on_lr = None,
        progress_bar = False,
        return_lr = False
):
    train_x, train_y = train_ds
    valid_x, valid_y = valid_ds
    perf_history = []
    lr_history = []
    epoch_range = tq.tqdm(range(epochs)) if progress_bar else range(epochs)
    for epoch in epoch_range:
        
        model.train()
        
        for i in range(0, len(train_x), batch_size):
            end_i = min(i + batch_size, len(train_x))
            xb = train_x[i : end_i]
            yb = train_y[i : end_i]
            loss_batch(model, loss_func, xb, yb, opt)

        
        model.eval()

        with t.no_grad():
            val_loss, nums = loss_batch(model,
                                        loss_func,
                                        valid_x,
                                        valid_y)

            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)

            perf_history.append(val_loss)
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
