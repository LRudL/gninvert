import torch as t
import torch_geometric as ptgeo

class LinearGNN(ptgeo.nn.MessagePassing):
    def __init__(self, node_features):
        super().__init__(aggr='add')
        self.m1 = t.nn.Linear(node_features * 2, node_features)
        self.u1 = t.nn.Linear(node_features * 2, node_features)
        self.num_node_features = node_features
    
    def message(self, x_i, x_j):
        inputs = t.cat([x_i, x_j], 1)
        # ^ [[x_i[0], x_j[0]], ...]
        return self.m1(inputs)
    
    def update(self, aggr_out, x=None):
        assert x is not None
        # aggr_out is the result of applying
        # the aggregation function (specified in the aggr
        # argument to super()) to all of the incoming messages
        inputs = t.cat([x, aggr_out], 1)
        return self.u1(inputs)
    
    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)
