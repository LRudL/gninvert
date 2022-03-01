import random
from gninvert.functions import run_GN
import torch as t
import torch_geometric as ptgeo

class TrainingData():
    def __init__(self, xs, ys, train_fraction=0.75, shuffle=True):
        if shuffle:
            seed = random.randint(0, 100)
            random.seed(seed)
            random.shuffle(xs)
            random.seed(seed)
            random.shuffle(ys)
            random.seed()
        self.xs = xs
        self.ys = ys
        train_i = round(len(xs) * train_fraction)
        self.train_x = xs[0:train_i]
        self.valid_x = xs[train_i:]
        self.train_y = ys[0:train_i]
        self.valid_y = ys[train_i:]

    def train_ds(self):
        return (self.train_x, self.train_y)

    def valid_ds(self):
        return (self.valid_x, self.valid_y)

    def train_size(self):
        return len(self.train_x)

    def valid_size(self):
        return len(self.valid_x)
        

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
    froms = t.tensor(froms)
    tos = t.tensor(tos)
    return t.stack([froms, tos], dim=0)

def generate_graphs_from_connections(edge_index, node_feature_num, num=10):
    """Takes an edge index and returns graphs that have that edge structure,
    but with nodes getting random values"""
    nodes = t.unique(edge_index.flatten())
    graphs = []
    uniform = t.distributions.uniform.Uniform(0, 1)
    for _ in range(num):
        x = uniform.sample((nodes.size()[0], node_feature_num))
        graphs.append(ptgeo.data.Data(edge_index=edge_index, x=x))
    return graphs

def generate_samples_from_graph(gn, iterations, gdata):
    """Generates pairs of [graph before gn step, graph after gn step]"""
    data_at_t = []
    def log_data(iteration_number, iteration_gdata):
        data_at_t.append(iteration_gdata)
    run_GN(gn, iterations, gdata, log_data)
    return (data_at_t[0:-1], data_at_t[1:])

g_edge_index1 = t.tensor(
    [[0, 1, 1, 2, 0, 2, 2, 3, 3, 4],
     [1, 0, 2, 1, 2, 0, 3, 2, 4, 3]],
    dtype=t.long)

g_edge_index2 = t.tensor(
    [[0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
     [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]],
    dtype=t.long)

g_edge_index3 = t.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1],
     [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]],
    dtype=t.long)

def generate_training_data(gn, edge_indices=None, graphs_per_edge_index=30, steps_per_graph=4):
    """Generate training data based on a GN"""
    if edge_indices == None:
        edge_indices = [g_edge_index1, g_edge_index2, g_edge_index3]
    graphs = []
    for edge_index in edge_indices:
        graphs += generate_graphs_from_connections(
            edge_index,
            gn.node_features,
            num=graphs_per_edge_index
        )
    x_train = []
    y_train = []
    for graph in graphs:
        (x_train2, y_train2) = generate_samples_from_graph(gn, steps_per_graph, graph)
        x_train += x_train2
        y_train += y_train2
    return (x_train, y_train)

def get_TrainingData(
        gn,
        train_fraction = 0.75,
        shuffle=True,
        edge_indices=None,
        graphs_per_edge_index=50,
        steps_per_graph=4
):
    """Returns a TrainingData object generated based on the GN provided""" 
    xs, ys = generate_training_data(gn, edge_indices, graphs_per_edge_index, steps_per_graph)
    return TrainingData(xs, ys, shuffle=shuffle, train_fraction=train_fraction)
