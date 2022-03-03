import sys
sys.path.append("..")

from gninvert.hyperparamsearch import hpsearch
from gninvert.gnns import LinearGNN, GNN_3Layer
from gninvert.gns import SingleDiffusionGN, MultiDiffusionGN, ActivatorInhibitorGN, EquationGN
from gninvert.graph_compare import model_steps_compare
from gninvert.data_generation import generate_graphs_from_connections, get_TrainingData
import torch as t
import argparse

all_args = argparse.ArgumentParser()

all_args.add_argument("-p", "--params", required=True,
                      help="file path for parameter settings")

all_args.add_argument("-m", "--model", required=True,
                      help="string for which GNN model to use")

all_args.add_argument("-g", "--gn", required=True,
                      help="which GN to use to generate the data")

all_args.add_argument("-o", "--output", required=True,
                      help="where to save the results")

args = vars(all_args.parse_args())

parameters = t.load(args['params'])

print("Running on parameters:")
print(parameters)

gns = {
    'SingleDiffusionGN': SingleDiffusionGN(diffusion_constant=0.1),
    'MultiDiffusionGN': MultiDiffusionGN(diffusion_constants=[0.1, 0.1]),
    'ActivatorInhibitorGN': ActivatorInhibitorGN(
        act_diff_const = 0.16,
        inh_diff_const = 0.10,
        growth_const = 0.2
    )
}

gn = gns[args['gn']]

if gn == None:
    raise Exception(f"Invalid GN: {args['gn']}")

models = {
    'linear': lambda : LinearGNN(node_features=gn.node_features),
    '3layer': lambda : GNN_3Layer(
        node_features = gn.node_features,
        message_features = gn.message_features,
        hidden_size = 10,
        final_gelu = True
    ),
    '3layer_geluless': lambda : GNN_3Layer(
        node_features = gn.node_features,
        message_features = gn.message_features,
        hidden_size = 10,
        final_gelu = False
    )
}

model_constructor = models[args['model']]

if model_constructor  == None:
    raise Exception(f"Invalid model: {args['model']}")

g_edge_index = t.tensor(
    [[0, 1, 1, 2, 0, 2, 2, 3, 3, 4],
     [1, 0, 2, 1, 2, 0, 3, 2, 4, 3]],
    dtype=t.long)
gdata = generate_graphs_from_connections(g_edge_index, gn.node_features, num=1)[0]

print("Started!")

results = hpsearch(
    parameters,
    model_constructor,
    lambda model : model_steps_compare(model, gn, gdata, iterations=10)['absolute']['avg_difs'][-1],
    get_TrainingData(gn)
)

print("Finished!")

t.save(results, args['output'])
