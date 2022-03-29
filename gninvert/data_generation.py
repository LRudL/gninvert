import random
from gninvert.functions import run_GN
import torch as t
import torch_geometric as ptgeo
from functools import reduce

def generic_shuffle(thing, things=None, seed=None):
    """
    Returns a shuffled version of `thing`, where `thing` is either a Python list or a tensor.
    Optionally takes `seed`, which can be used for replicability.
    Optionally takes `things`, a list/tensor of the same size as `thing`,
    which should be shuffled using the same permutation as thing.
    If `things` is passed, returns a list, where shuffled `thing` is the first element and
    the other shuffled `things` follow.
    Does not mutate anything.
    `thing` must be of the same type as every element in `things`.
    """
    if seed == None:
        seed = random.randint(0, 1000)
    if type(thing) is list:
        shuffled = thing.copy()
        random.seed(seed)
        random.shuffle(shuffled)
    elif t.is_tensor(thing):
        shuffled = thing.detach().clone()
        t.manual_seed(seed)
        shuffled = shuffled[t.randperm(thing.shape[0])]
    else:
        raise Exception(f"generic_shuffle cannot shuffle: {thing}")
    if things != None:
        for ti in things:
            if type(ti) != type(thing):
                raise Exception("Cannot generic_shuffle a mix of tensors and lists.")
        shuffled_things = [generic_shuffle(ti, seed=seed) for ti in things]
        return [shuffled] + shuffled_things
    return shuffled

def listlike_equals(ls1, ls2):
    if t.is_tensor(ls1) and t.is_tensor(ls2):
        return t.equal(ls1, ls2)
    if (type(ls1) == list or type(ls1) == tuple) and (type(ls2) == list or type(ls2) == tuple):
        if len(ls1) != len(ls2):
            return False
        if len(ls1) == 0 and len(ls2) == 0:
            return True
        return reduce(lambda a, b: a and b,
                      [listlike_equals(item1, item2) for (item1, item2) in zip(ls1, ls2)])
    return ls1 == ls2

class TrainingData():
    def __init__(self, xs, ys, train_fraction=0.75, shuffle=True, shuffle_seed = None):
        if shuffle:
            if shuffle_seed == None:
                shuffle_seed = random.randint(0, 1000)
            xs, ys = generic_shuffle(xs, [ys], shuffle_seed)
        self.shuffled = shuffle
        self.shuffle_seed = shuffle_seed
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

    def __repr__(self):
        return f"<gninvert.data_generation.TrainingData, {self.train_size()} train size / \
{self.valid_size()} validation size / shuffle {self.shuffled}>"

    def __eq__(self, other):
        if type(other) == type(self):
            train_equal = listlike_equals(self.train_ds(), other.train_ds())
            valid_equal = listlike_equals(self.valid_ds(), other.valid_ds())
            return train_equal and valid_equal
        return super.__eq__(self, other)

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
