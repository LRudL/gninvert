import numpy as np
import math
from functools import reduce
import networkx as nx

class Attribute:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __repr__(self):
        return f"<ATTRIBUTE {self.name} taking values {self.values}"

class Example:
    def __init__(self, attribute_vals, value):
        self.attr_values = attribute_vals
        self.val = value

    def __repr__(self):
        return f"<EXAMPLE: val {self.val}, attributes: {self.attr_values}"

class LeafNode:
    def __init__(self, val):
        self.val = val
        self.leaf = True

    def __repr__(self):
        return f"<LEAF {self.val}>"

def indent_text(s, size=2):
    return s.replace("\n", "\n" + " "*size)

class Node:
    def __init__(self, test_attr):
        self.test_attribute = test_attr
        self.leaf = False
        self.children = dict()

    def add_child(self, test_attr_val, subtree):
        if type(test_attr_val) == list:
            test_attr_val = tuple(test_attr_val)
        self.children[test_attr_val] = subtree

    def __repr__(self):
        s = f"<<<DTREE on {self.test_attribute}"
        for val, subtree in self.children.items():
            s += f"\n  --({val})-->\n  {indent_text(subtree.__repr__())}"
        s += " >>>"
        return s

def argmax(args, fn):
    max_val = float('-inf')
    max_arg = None
    for arg in args:
        val = fn(arg)
        if val > max_val:
            max_val = val
            max_arg = arg
    return max_arg
        
def log_based_representative_value(values):
    logsum = 0
    for val in values:
        logsum += math.log(val)
    return logsum / len(values)

def variance(vals):
    return np.var(np.array(vals))

def approx_equal_gen(threshold):
    def f(vals):
        return math.log(max(vals)) - math.log(min(vals)) < threshold
    return f

def split_on(getter, examples):
    split_dict = {}
    for ex in examples:
        prop = getter(ex)
        if type(prop) == list:
            prop = tuple(prop)
        if prop in split_dict.keys():
            split_dict[prop] = tuple(list(split_dict[prop]) + [ex])
        else:
            split_dict[prop] = tuple([ex])
    return split_dict

def log_variance_reduction_importance(attribute, examples):
    prev_variance = variance([example.val for example in examples])
    split_dict = split_on(lambda example : example.attr_values[attribute.name],
                          examples)
    total_var = 0
    for attr_val in split_dict.keys():
        total_var += variance([math.log(example.val) for example in list(split_dict[attr_val])])
    mean_var = total_var / len(list(split_dict.keys()))
    return prev_variance - mean_var

def decision_tree(
        examples, attributes, parent_examples,
        importance_fn, representative_value_fn, equality_fn
):
    example_vals = [example.val for example in examples]
    parent_vals = [example.val for example in parent_examples]
    if len(examples) == 0:
        return LeafNode(representative_value_fn(parent_vals))
    elif equality_fn(example_vals):
        return LeafNode(representative_value_fn(example_vals))
    elif len(attributes) == 0:
        return LeafNode(representative_value_fn(parent_vals))
    else:
        best_attr = argmax(attributes, lambda attr : importance_fn(attr, examples))
        assert type(best_attr) == Attribute
        tree = Node(best_attr)
        for val in best_attr.values:
            subexamples = {ex for ex in examples if ex.attr_values[best_attr.name] == val}
            subtree = decision_tree(
                subexamples, attributes - set([best_attr]), examples,
                importance_fn, representative_value_fn, equality_fn
            )
            tree.add_child(val, subtree)
        return tree

def decision_tree_paths(dt, sort=True):
    if dt.leaf:
        return [[dt.val]]
    else:
        paths = []
        for dec in dt.children.keys():
            subpaths = decision_tree_paths(dt.children[dec])
            paths += [[(dt.test_attribute.name, dec)] + subpath for subpath in subpaths]
    if sort:
        paths = sorted(paths, key = lambda path : path[-1])
    return paths

def dtree_to_networkx(dt):
    def node_adder(g, dt):
        g.add_node(dt)
        if not dt.leaf:
            for attr_val, child in dt.children.items():
                g, node_for_child = node_adder(g, child)
                g.add_edge(dt, node_for_child, label=attr_val)
        return g, dt
    g = nx.DiGraph()
    return node_adder(g, dt)[0]

def draw_dtree(dt):
    nxg = dtree_to_networkx(dt)
    node_labels = {}
    for node in nxg.nodes():
        node_labels[node] = node.test_attribute.name if not node.leaf else round(node.val, 1)
    edge_labels = {}
    for node1, node2 in nxg.edges():
        edge_labels[(node1, node2)] = nxg.get_edge_data(node1, node2)['label']
    pos = nx.shell_layout(nxg)
    leaf_nodes = [x for x in nxg.nodes() if nxg.out_degree(x)==0 and nxg.in_degree(x)==1]
    leaf_vals = [x.val for x in leaf_nodes]
    leaf_cols = [((v - min(leaf_vals)) / (max(leaf_vals) - min(leaf_vals)), 0, 0) for v in leaf_vals]
    nx.draw_networkx(nxg, pos, with_labels=False)
    nx.draw_networkx_nodes(nxg, pos, nodelist=leaf_nodes, node_color = leaf_cols)
    nx.draw_networkx_labels(nxg, pos, labels=node_labels, verticalalignment='bottom')
    nx.draw_networkx_edge_labels(nxg, pos, edge_labels=edge_labels)
