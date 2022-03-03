import torch as t
import torch_geometric as ptgeo

class SingleDiffusionGN(ptgeo.nn.MessagePassing):
    def __init__(self, diffusion_constant):
        super().__init__(aggr='add')
        self.diffusion_constant = diffusion_constant
        self.node_features = 1
        self.message_features = 1
    
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
        self.message_features = self.node_features
    
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
        self.node_features = 3
        self.message_features = 2

    def message(self, x_i, x_j):
        return self.diff_consts * (x_j[:, 1:] - x_i[:, 1:])

    def update(self, aggr_out, x):
        # give things names for convenience:
        vol = x[:, 0] # cell sizes
        act_conc = x[:, 1] # activator concentration
        inh_conc = x[:, 2] # inhibitor concentration
        
        growth = ((act_conc - inh_conc) / vol * self.growth_const)
        
        act_inh_delta_vec = aggr_out # 
        
        delta_vec = t.cat([growth.unsqueeze(1), act_inh_delta_vec], dim=1)
        
        result = x + delta_vec
        
        return result

    def forward(self, gdata):
        return self.propagate(gdata.edge_index, x=gdata.x)

class FullActInhGN(ptgeo.nn.MessagePassing):
    def __init__(self,
                 spatial_const,
                 temporal_const,
                 growth_alpha,
                 growth_rho,
                 growth_scale,
                 reaction_const,
                 reference_conc = 1
    ):
        super().__init__(aggr='add')
        self.spatial_const = spatial_const
        self.temporal_const = temporal_const
        self.growth_alpha = growth_alpha
        self.growth_rho = growth_rho
        self.growth_scale = growth_scale
        self.reaction_const = reaction_const
        self.reference_conc = reference_conc
        self.eps = 0.01
        self.node_features = 3
        self.message_features = 2

    def message(self, x_i, x_j):
        consts = t.tensor([
            self.spatial_const * self.temporal_const,
            self.temporal_const
        ])
        return consts * (x_j[:, 1:] - x_i[:, 1:])

    def update(self, aggr_out, x):
        # give things names for convenience:
        vol = x[:, 0] # cell sizes
        act_conc = x[:, 1] # activator concentration
        inh_conc = x[:, 2] # inhibitor concentration

        g = (act_conc / vol)**self.growth_alpha
        growth = g / (self.growth_rho**self.growth_alpha + g) * self.growth_scale

        delta_act = aggr_out[:, 0]
        delta_act += (act_conc**2) / (inh_conc + self.eps) - act_conc
        delta_act *= self.reaction_const

        delta_inh = aggr_out[:, 1]
        delta_inh += (act_conc**2) / (self.reference_conc * vol + self.eps) - inh_conc
        delta_inh *= self.reaction_const
        
        delta_vec = t.cat([
            growth.unsqueeze(1),
            delta_act.unsqueeze(1),
            delta_inh.unsqueeze(1)
        ], dim=1)
        
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
