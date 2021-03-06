# gninvert

The purpose of this project is to explore (and develop tools for) the process of "inverting" graph networks (GNs): rather than using a GN to simulate or make predictions, the goal is to find a GN with interpretable symbolic message and update functions that might have generated a given data set.

This is done in two ways:
- First, train a graph neural network (GNN), i.e. make its message and update functions neural networks and train them to approximate the behaviour of certain time series data
- Second, take the GNN, and use symbolic regression to fit equations to what it has learned. This gives a GN with interpretable functions, rather than a black-box GNN.

The method is based on [this paper](https://arxiv.org/abs/2006.11287).

In addition to the inversion, the project contains code for several GN-based simulations, which generate the data. These simulations are inspired by simplified versions of the cell morphogenesis models in [this paper](https://pubmed.ncbi.nlm.nih.gov/29402913/). Cell morphogenesis models are the main application area considered in this project.


## Setup & Dependencies

First, you need to install the programming language Julia, required by PySR (which is a Python library, like all of gninvert, but internally uses Julia).

Second, you must use Pyenv as your Python environment manager because otherwise PySR will throw a fit and probably need to reinstall itself on every run. Specifically, you need to create a Pyenv environment by doing something like this: `PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.10`.

After these steps, and making sure that Pyenv is working (`which pip` and `which python` should both show non-default values, specifically paths that include words like `.pyenv` and/or `shims` in them), you can simple run `chmod +x dependecy_install.sh` and `./dependency_install.sh` in succession in a terminal in the root `gninvert` folder and this will install the Python packages for you (and configure PySR).

The packages installed are:

- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [gplearn](https://gplearn.readthedocs.io/en/stable/installation.html)
- NumPy
- Pandas
- Jupyter Lab (only required for notebook-related stuff)
- ipywidgets (only required for notebook-related stuff)
- tqdm (only required for notebook-related stuff)
- matplotlib
- networkx
- einops
- PySR


## Example usage

The following command, run in the project directory, will do a hyperparameter search using the "general" hyperparameter settings over data from the 'diff2' model (diffusion network with two chemicals), running each hyperparameter setting 5 times, and saving the results:

```
python run.py -m diff2 -p general -r 5 -n name_to_save_under
```

The file `run.py` lists available models and hyperparameter options, in a format that is easy to extend yourself.

Other options available:

- pass `-s 1` to skip the symbolic regression and only do the hyperparameter search
- pass `-S 1` to only do symbolic regression (on the model specified using `-m` directly, rather than on data generated from that model), skipping the hyperparameter search and the GNN training
- pass `-c simulation` or `-c loss` to change whether the best GNN model is selected based on how accurately it simulates the original GN model, or on last-step validation loss in the training loop.
