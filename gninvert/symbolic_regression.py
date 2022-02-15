import torch as t
from gplearn.genetic import SymbolicRegressor

def get_pysr_equations(
        dimensions, function, n=200,
        niterations = 5,
        variable_names = None,
        constraints = None
):
    input_var_blocks = [
        t.rand((n, dimension))
        for dimension in dimensions
    ]
    input_vars_concat = t.cat(input_var_blocks, dim=1).detach().numpy()
    outputs = function(*input_var_blocks).detach().numpy()
    model = PySRRegressor(
        niterations=niterations,
        variable_names=variable_names,
        constraints=constraints,
        multithreading=False
    )
    X = input_vars_concat.detach().numpy()
    Y = outputs.detach().numpy()
    model.fit(X, Y)
    return model


def get_gplearn_equations(
    dimensions, function, n=200,
):
    input_var_blocks = [
        t.rand((n, dimension))
        for dimension in dimensions
    ]
    input_vars_concat = t.cat(input_var_blocks, dim=1).detach().numpy()
    outputs = function(*input_var_blocks).detach().numpy()
    # assumption: outputs is of shape [n_examples, dimensionality_of_output]
    assert len(outputs.shape) == 2
    srs = []
    for i in range(outputs.shape[-1]):
        sr = SymbolicRegressor(population_size=100000,
                               generations=30,
                               stopping_criteria=0.001,
                               tournament_size=50,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9,
                               parsimony_coefficient=0.01,
                               verbose=1)
        sr.fit(input_vars_concat, outputs[:, i])
        srs.append(sr)
    return srs
