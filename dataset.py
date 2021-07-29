import os.path as osp
import re
# from feature_expansion import FeatureExpander
from torch_geometric.datasets import TUDataset


def get_dataset(name, root=None, use_node_attr=False):
    if root is None:
        root = '../data/' + name
    dataset = TUDataset(
        root, name, pre_transform=None, use_node_attr=use_node_attr
    )
    return dataset
