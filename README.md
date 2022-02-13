# gninvert

The purpose of this project is to explore (and develop tools for) the process of "inverting" graph networks (GNs): rather than generating data from a GN, we take data and try to figure out what GN produced it. Since a GN is parametrised by the message and update functions, this means finding those.

This is done in two ways:
- First, train a graph neural network (GNN), i.e. make its message and update functions neural networks and train them to approximate the behaviour of certain time series data
- Second, take the GNN, and use symbolic regression to fit equations to what it has learned. This gives a GN with interpretable functions, rather than a black-box GNN.

The method is based on [this paper](https://arxiv.org/abs/2006.11287).

In addition to the inversion, the project contains code for several GN-based simulations, which generate the data. These simulations are inspired by simplified versions of the cell morphogenesis models in [this paper](https://pubmed.ncbi.nlm.nih.gov/29402913/). Cell morphogenesis models are the main application area considered in this project.


## Dependencies
[PyTorch](https://pytorch.org/get-started/locally/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [gplearn](https://gplearn.readthedocs.io/en/stable/installation.html), NumPy, Jupyter Lab (to run notebooks), tqdm (+ ipywidgets to work in Jupyter), matplotlib, networkx
