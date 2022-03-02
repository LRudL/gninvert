import torch as t
import torch_geometric as ptgeo
from gninvert.functions import gn_time_series
import matplotlib.pyplot as plt
#import networkx as nx
#from ptgeo.utils import to_networkx, from_networkx

from gninvert.functions import gdisplay

def graph_compare(
        gdata1, gdata2,
        verbose = False, tol = 0.01,
        eps = 1e-7,
        return_value = False,
        absolute = True
):
    edges1 = gdata1.edge_index.shape[1]
    edges2 = gdata2.edge_index.shape[1]
    if edges1 != edges2:
        print("Graphs have non-equal edge numbers: {edges1} and {edges2}")
        return False
    dif = t.abs(gdata1.x - gdata2.x)
    if not absolute:
        dif = t.abs(gdata1.x) / (t.abs(gdata2.x) + eps)
        for x in range(gdata1.x.shape[0]): # horror code, make more torch-like
            for y in range(gdata2.x.shape[1]):
                if t.abs(gdata1.x[x][y]) < t.abs(gdata2.x[x][y]):
                    dif[x][y] = t.abs(gdata2.x[x][y]) / (t.abs(gdata1.x[x][y]) + eps)
        
    if verbose:
        print("Differences tensor:")
        print(dif)
    max_dif = t.max(dif).item()
    avg_dif = t.mean(dif).item()
    min_dif = t.min(dif).item()
    if verbose:
        if absolute:
            print(f"Max/avg/min distance: {max_dif}, {avg_dif}, {min_dif}")
        else:
            print(f"Maximum relative difference: {max_dif*100}%")
            print(f"Average relative difference: {avg_dif * 100}%")
            print(f"Minimum relative difference: {min_dif * 100}%")
    return (max_dif, avg_dif, min_dif) if return_value else max_dif < tol

g_edge_index = t.tensor(
    [[0, 1, 1, 2, 0, 2, 2, 3, 3, 4],
     [1, 0, 2, 1, 2, 0, 3, 2, 4, 3]],
    dtype=t.long
)
g_x = t.tensor([[0.0, 0.0],
                [0.7, 0.0],
                [0.4, 0.0],
                [0.9, 0.0],
                [0.0, 1.0]],
               dtype=t.float)
g_data = ptgeo.data.Data(x=g_x, edge_index=g_edge_index)

def model_steps_compare(model, base, gdata=None, iterations=20):
    if gdata is None:
        gdata = g_data
    model_steps = gn_time_series(model, iterations, gdata)[1:]
    base_steps = gn_time_series(base, iterations, gdata)[1:]
    rel_dif_stats = [
        graph_compare(gdata2, gdata1, return_value = True, absolute = False)
        for (gdata1, gdata2) in zip(model_steps, base_steps)
    ]
    abs_dif_stats = [
        graph_compare(gdata2, gdata1, return_value = True, absolute = True)
        for (gdata1, gdata2) in zip(model_steps, base_steps)
    ]
    return {
        "model_steps" : model_steps,
        "base_steps" : base_steps,
        "relative" : {
            "max_difs" : [t[0] for t in rel_dif_stats],
            "avg_difs" : [t[1] for t in rel_dif_stats],
            "min_difs" : [t[2] for t in rel_dif_stats]
        },
        "absolute" : {
            "max_difs" : [t[0] for t in abs_dif_stats],
            "avg_difs" : [t[1] for t in abs_dif_stats],
            "min_difs" : [t[2] for t in abs_dif_stats]
        },
    }

def model_compare(model, base, gdata=None, iterations=20):
    if gdata == None:
        gdata = g_data
    info = model_steps_compare(model, base, gdata, iterations)
    gdata_pairs = list(zip(info["model_steps"], info["base_steps"]))
    max_rel_difs = info["relative"]["max_difs"]
    avg_rel_difs = info["relative"]["avg_difs"]
    min_rel_difs = info["relative"]["min_difs"]

    max_abs_difs = info["absolute"]["max_difs"]
    avg_abs_difs = info["absolute"]["avg_difs"]
    min_abs_difs = info["absolute"]["min_difs"]
    
    print(f"Stats for {iterations} steps for node features:")
    
    print(f"Greatest relative difference: {max(max_rel_difs)*100}%")
    print(f"Average relative difference: {sum(avg_rel_difs) / iterations * 100}%")
    print(f"Smallest relative difference: {min(min_rel_difs)*100}%")

    print(f"Greatest absolute difference: {max(max_abs_difs)}")
    print(f"Average absolute difference: {sum(avg_abs_difs) / iterations}")
    print(f"Smallest absolute difference: {min(min_abs_difs)}%")

    # graph relative differences
    fig, ax = plt.subplots(1, 1)
    ax.plot(max_rel_difs, "-o", label="Max rel. diffs.")
    ax.plot(avg_rel_difs, "-o", label="Avg rel. diffs.")
    ax.plot(min_rel_difs, "-o", label="Min rel. diffs.")
    ax.set_ylim([1.0, 2.5])
    plt.title("Max/avg/min relative differences over run")
    plt.legend()
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(max_abs_difs, "-o", label="Max abs. diffs.")
    ax.plot(avg_abs_difs, "-o", label="Avg abs. diffs.")
    ax.plot(min_abs_difs, "-o", label="Min abs. diffs.")
    plt.title("Max/avg/min absolute differences over run")
    plt.xlabel("Iterations")
    plt.ylabel("Absolute feature value difference ")
    plt.legend()
    plt.show()

    # visualize final graphs
    print(f"The model being tested finished the run outputting this graph:")
    gdisplay(gdata_pairs[-1][0])
    print(gdata_pairs[-1][0].x)
    print(f"The ground truth model finished the run outputting this graph:")
    gdisplay(gdata_pairs[-1][1])
    print(gdata_pairs[-1][1].x)
    
    # conservation laws check
    model_sums = [t.sum(g_pair[0].x).item() for g_pair in gdata_pairs]
    base_sums = [t.sum(g_pair[1].x).item() for g_pair in gdata_pairs]
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(model_sums, "-+", label="model feature sum")
    ax2.plot(base_sums, "-+", label="base feature sum")
    plt.title("Conservation law?")
    plt.xlabel("Iterations")
    plt.ylabel("Sum of feature values")
    plt.legend()
    plt.show()
    
    return (gdata_pairs[-1][0], gdata_pairs[-1][1])
