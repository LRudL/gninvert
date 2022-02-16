import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import torch
import torch_geometric as ptgeo
from einops import rearrange, reduce, repeat
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import time

def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin


def run_GN(
        gn, iterations, data,
        log_fn = lambda i, x : i
):
    """Run a graph network gn for some number of iterations, starting with
    data (a PtGeo data object) data (i.e. has a .x attribute for node features
    and a .edge_index attribute for edge features, of the form
    [[u1, u2, ...], [v1, v2, ...]] for edges [u1,v1], [u2,v2], ...)"""
    log_fn(0, data)
    node_attr_tensor = data.x
    edge_index = data.edge_index
    for i in range(iterations):
        node_attr_tensor = gn.forward(data)
        data = ptgeo.data.Data(x=node_attr_tensor, edge_index=edge_index)
        log_fn(i+1, data)
    return ptgeo.data.Data(x=node_attr_tensor,
                           edge_index=edge_index)
def gn_time_series(gn, iterations, data):
    gdata_time_series = []
    def log_graph(i, gdata):
        gdata_time_series.append(gdata)
    run_GN(gn, iterations, data, log_fn = log_graph)
    return gdata_time_series

def make_color_scale(minimum, maximum, color_map=plt.cm.plasma):
    """Return a color scale function that maps values between
    minimum and maximum to colors"""
    def get_color(color_map, val):
        return pltcolors.rgb2hex(color_map(val))
    def normed(val):
        return (val - minimum) / (maximum - minimum)
    return lambda x : get_color(color_map, normed(x))


def gdisplay(gdata, color_scales = None):
    """Display a graph from PtGeo graph data type, with some color scale for the nodes."""
    nx_graph = ptgeo.utils.to_networkx(gdata)
    node_features = gdata.x[0].shape[0]
    if color_scales == None:
        color_scales = [make_color_scale(0, 1) for _ in range(node_features)]
    fig, ax = plt.subplots(1, node_features)
    if node_features == 1:
        ax = [ax]
    for i in range(node_features):
        node_colors = list(map(color_scales[i],
                               torch.transpose(gdata.x, 0, 1)[i].tolist()))
        pos = nx.spring_layout(nx_graph, seed=42)
        nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, ax=ax[i])
        nx.draw_networkx_edges(nx_graph, pos, ax=ax[i], arrows=False)
    plt.show()

def run_and_draw(gn, gdata, iterations, draw_interval=3, color_scales = None, log=True):
    """Like run_GN, except draws the GN at a certain interval of steps."""
    def drawer(iteration, data):
        s = sum(data.x.flatten())
        if iteration % draw_interval == 0:
            if log:
                print(f"Iteration {iteration}")
                print(f"sum: {s} / node values:")
                print(data.x.tolist())
            gdisplay(data, color_scales)
    return run_GN(gn, iterations, gdata, drawer)


## GRAPH GENERATION

def generate_grid_edge_index(n):
    froms = []
    tos = []
    for i in range(n*n):
        pos = i % n
        row = i // n
        if pos < n - 1:
            froms.append(i)
            tos.append(i + 1)
        if pos > 0:
            froms.append(i)
            tos.append(i - 1)
        if row < n - 1:
            froms.append(i)
            tos.append(i + n)
        if row > 0:
            froms.append(i)
            tos.append(i - n)
    froms = torch.tensor(froms)
    tos = torch.tensor(tos)
    return torch.stack([froms, tos], dim=0)


## TRAINING DATA GENERATION


def generate_graphs_from_connections(edge_index, node_feature_num, num=10):
    """Takes an edge index and returns graphs that have that edge structure,
    but with nodes getting random values"""
    nodes = torch.unique(edge_index.flatten())
    graphs = []
    for _ in range(num):
        x = torch.nn.Sigmoid()(torch.randn(nodes.size()[0], node_feature_num))
        graphs.append(ptgeo.data.Data(edge_index=edge_index, x=x))
    return graphs

def generate_samples_from_graph(gn, iterations, gdata):
    """Generates pairs of [graph before gn step, graph after gn step]"""
    data_at_t = []
    def log_data(iteration_number, iteration_gdata):
        data_at_t.append(iteration_gdata)
    run_GN(gn, iterations, gdata, log_data)
    return (data_at_t[0:-1], data_at_t[1:])

def generate_training_data(gn):
    """Generate training data based on an arbitrary hard-coded graph"""
    g_edge_index = torch.tensor(
        [[0, 1, 1, 2, 0, 2, 2, 3, 3, 4],
         [1, 0, 2, 1, 2, 0, 3, 2, 4, 3]],
        dtype=torch.long)
    
    edge_indices = [g_edge_index]
    graphs = []
    for edge_index in edge_indices:
        graphs += generate_graphs_from_connections(edge_index, gn.node_features)
    x_train = []
    y_train = []
    for graph in graphs:
        (x_train2, y_train2) = generate_samples_from_graph(gn, 20, graph)
        x_train += x_train2
        y_train += y_train2
    return (x_train, y_train)


## TRAINING
## Code below is DEPRECATED
## (see training.py for more up-to-date training code)

def graphs_loss_func(model, xb, yb):
    node_loss_func = torch.nn.MSELoss()
    # result = list(map(run_model, xb))
    result = [model(gdata) for gdata in xb]
    losses = [node_loss_func(node_results, ygdata.x)
              for (node_results, ygdata) in
              zip(result, yb)]
    return sum(losses)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model, xb, yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # loss.item(): pure numeric value of loss
    # also returns length of the batch, 
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_ds, valid_ds, batch_size=10):
    train_x, train_y = train_ds
    valid_x, valid_y = valid_ds
    perf_history = []
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_x), batch_size):
            end_i = min(i + batch_size, len(train_x))
            xb = train_x[i : end_i]
            yb = train_y[i : end_i]
            loss_batch(model, loss_func, xb, yb, opt)
        
        model.eval()
        with torch.no_grad():
            val_loss, nums = loss_batch(model,
                                        loss_func,
                                        valid_x,
                                        valid_y)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(epoch, val_loss)
