import torch as t
import torch_geometric as ptgeo
from gninvert.nn import GeneralLinearFullNet

class LinearGNN(ptgeo.nn.MessagePassing):
    def __init__(self, node_features):
        super().__init__(aggr='add')
        self.m1 = t.nn.Linear(node_features * 2, node_features)
        self.u1 = t.nn.Linear(node_features * 2, node_features)
        self.node_features = node_features
    
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

    
class GNN_3Layer(ptgeo.nn.MessagePassing):
    def __init__(self, node_features, message_features=None, hidden_size=6, final_gelu = True):
        super().__init__(aggr='add')
        if message_features == None:
            message_features = node_features
        #self.m = t.nn.Sequential(
        #t.nn.Linear(node_features * 2, hidden_size),
        #t.nn.GELU(),
        #    t.nn.Linear(hidden_size, message_features)
        #)
        #if final_gelu:
        #    self.m = t.nn.Sequential(self.m, t.nn.GELU())
        #self.u = t.nn.Sequential(
        #    t.nn.Linear(node_features + message_features, hidden_size),
        #    t.nn.GELU(),
        #    t.nn.Linear(hidden_size, node_features)
        #)
        #if final_gelu:
        #    self.u = t.nn.Sequential(self.u, t.nn.GELU())

        self.m = GeneralLinearFullNet(
            in_features = node_features * 2,
            out_features = message_features,
            hidden_sizes = [hidden_size],
            nonlinearity = t.nn.GELU(),
            end_with_nonlinearity = final_gelu
        )
        self.u = GeneralLinearFullNet(
            in_features = node_features + message_features,
            out_features = node_features,
            hidden_sizes = [hidden_size],
            nonlinearity = t.nn.GELU(),
            end_with_nonlineraity = final_gelu
        )
        self.node_features = node_features
        self.message_features = message_features
        self.hidden_size = hidden_size
        self.final_gelu = final_gelu
    
    def message(self, x_i, x_j):
        inputs = t.cat([x_i, x_j], 1)
        # ^ [[x_i[0], x_j[0]], ...]
        return self.m(inputs)
    
    def update(self, aggr_out, x=None):
        assert x is not None
        # aggr_out is the result of applying
        # the aggregation function (specified in the aggr
        # argument to super()) to all of the incoming messages
        inputs = t.cat([x, aggr_out], 1)
        return self.u(inputs)
    
    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)


    
class GNN_full(ptgeo.nn.MessagePassing):
    def __init__(
            self,
            node_features, message_features,
            message_hidden_sizes = [64],
            message_nonlinearity = t.nn.GELU,
            message_end_with_nonlinearity = True,
            update_hidden_sizes = None,
            update_nonlinearity = None,
            update_end_with_nonlinearity = None
    ):
        super().__init__(aggr='add')
        if update_hidden_sizes == None and update_nonlinearity == None and update_end_with_nonlinearity == None:
            self.message_and_update_equal = True
        else:
            self.message_and_update_equal = False
        if update_hidden_sizes == None:
            update_hidden_sizes = message_hidden_sizes
        if update_nonlinearity == None:
            update_nonlinearity = message_nonlinearity
        if update_end_with_nonlinearity == None:
            update_end_with_nonlinearity = message_end_with_nonlinearity
        self.m = GeneralLinearFullNet(
            in_features = node_features * 2,
            out_features = message_features,
            hidden_sizes = message_hidden_sizes,
            nonlinearity = message_nonlinearity,
            end_with_nonlinearity = message_end_with_nonlinearity
        )
        self.u = GeneralLinearFullNet(
            in_features = node_features + message_features,
            out_features = node_features,
            hidden_sizes = update_hidden_sizes,
            nonlinearity = update_nonlinearity,
            end_with_nonlinearity = update_end_with_nonlinearity
        )
        self.node_features = node_features
        self.message_features = message_features
        self.message_hidden_size = message_hidden_sizes
        self.update_hidden_size = update_hidden_sizes
        self.message_nonlinearity = message_nonlinearity
        self.update_nonlinearity = update_nonlinearity
        self.message_end_with_nonlinearity = message_end_with_nonlinearity
        self.update_end_with_nonlinearity = update_end_with_nonlinearity

        self.message_hook = None
    
    def message(self, x_i, x_j):
        inputs = t.cat([x_i, x_j], 1)
        # ^ [[x_i[0], x_j[0]], ...]
        out = self.m(inputs)
        if hasattr(self, 'message_hook'):
            if self.message_hook != None:
                self.message_hook(out)
        return out
    
    def update(self, aggr_out, x=None):
        assert x is not None
        # aggr_out is the result of applying
        # the aggregation function (specified in the aggr
        # argument to super()) to all of the incoming messages
        inputs = t.cat([x, aggr_out], 1)
        return self.u(inputs)
    
    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)
