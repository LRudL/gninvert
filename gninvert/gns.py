import torch as t
import torch_geometric as ptgeo

class SingleDiffusionGN(ptgeo.nn.MessagePassing):
    def __init__(self, diffusion_constant):
        super().__init__(aggr='add')
        self.diffusion_constant = diffusion_constant
        self.node_features = 1
    
    def message(self, x_i, x_j):
        x_target = x_i
        x_source = x_j
        return self.diffusion_constant * (x_source - x_target) # try adding noise here
    
    def update(self, aggr_out, x=None):
        assert x is not None
        # aggr_out is the result of applying
        # the aggregation function (specified in the aggr
        # argument to super()) to all neighbours
        return x + aggr_out
    
    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)

    
class MultiDiffusionGN(ptgeo.nn.MessagePassing):
    def __init__(self, diffusion_constants):
        super().__init__(aggr='add')
        self.diffusion_constants = t.tensor(diffusion_constants)
        self.node_features = len(diffusion_constants)
    
    def message(self, x_i, x_j):
        return self.diffusion_constants * (x_j - x_i)
    
    def update(self, aggr_out, x=None):
        assert x is not None
        return x + aggr_out
    
    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)

class ActivatorInhibitorGN(ptgeo.nn.MessagePassing):
    def __init__(self, act_diff_const, inh_diff_const, growth_const):
        super().__init__(aggr='add')
        self.diff_consts = t.tensor([act_diff_const, inh_diff_const])
        self.act_diff_const = act_diff_const
        self.inh_diff_const = inh_diff_const
        self.growth_const = growth_const

    def message(self, x_i, x_j):
        return self.diff_consts * x_j[:, 1:] - x_i[:, 1:]

    def update(self, aggr_out, x):
        vol = x[:, 0]
        act_concentration = x[:, 1]
        inh_concentration = x[:, 2]
        growth = ((act_concentration - inh_concentration) / vol * self.growth_const).unsqueeze(0).squeeze(0)
        act_inh_delta_vec = aggr_out
        delta_vec = t.cat([growth.unsqueeze(1), act_inh_delta_vec], dim=1)
        result = x + delta_vec
        return result

    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)
        


def component_fns_to_vector_fn(component_fns):
    def fn(in_tensor):
        comps = [comp_fn(in_tensor) for comp_fn in component_fns]
        stacked = t.stack(comps, dim=1).clone().detach()
        return stacked
    return fn

class EquationGN(ptgeo.nn.MessagePassing):
    def __init__(self, message_fn, update_fn):
        super().__init__(aggr='add')
        if isinstance(message_fn, tuple) or isinstance(message_fn, list):
            message_fn = component_fns_to_vector_fn(message_fn)
        if isinstance(update_fn, tuple) or isinstance(update_fn, list):
            update_fn = component_fns_to_vector_fn(update_fn)
        self.message_fn = message_fn
        self.update_fn = update_fn

    def message(self, x_i, x_j):
        return self.message_fn(t.cat([x_i, x_j], 1))#.unsqueeze(dim=-1)

    def update(self, aggr_out, x=None):
        return self.update_fn(t.cat([aggr_out, x], 1))#.unsqueeze(dim=-1)

    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)
