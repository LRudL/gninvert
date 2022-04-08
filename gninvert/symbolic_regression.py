import torch as t
from gplearn.genetic import SymbolicRegressor
from pysr import PySRRegressor
import numpy as np

def get_pysr_equations(
        dimensions, function, n=1000,
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
        variable_names = variable_names,
        progress=True,
        verbosity=0
    )
    X = input_vars_concat
    Y = outputs
    model.fit(X, Y)
    return model

def get_best_eq(sr_obj):
    return [
        sr_obj.equations[i][sr_obj.equations[i].score == \
                            sr_obj.equations[i].score.max()
        ].iloc[0].sympy_format
        for i in range(len(sr_obj.equations))
    ]

def pysr_test():
    X = 4.321 * np.random.randn(100, 5)
    y = 1.234 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
    model = PySRRegressor(
        niterations=5,
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",  # Custom operator (julia syntax)
        ],
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
    )
    model.fit(X, y)
    return model

# prefer PySR to gplearn, but PySR is picky about its run environment,
# so if you have better things to do with your time than fiddle with
# pyenv, you can try using:
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
