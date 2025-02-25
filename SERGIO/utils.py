import numpy as np
import scanpy as sc
import pandas as pd
import csv
import networkx as nx
from SERGIO.sergio import sergio
import warnings


# from Sergio txt to networkx structure
def convert_interaction_net_to_networkx(input_file_taregts):
    """This function converts the input text file where interaction is stored into a networkx object"""
    G = nx.DiGraph()
    with open(input_file_taregts) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            unformatted = line[0].split(",")
            i = float(unformatted[0])
            in_degree = float(unformatted[1])
            if i.is_integer() & in_degree.is_integer():
                i = int(i)
                in_degree = int(in_degree)
            else:
                raise ValueError(
                    "1st and 2nd column of "
                    + input_file_taregts
                    + " should be integers, but they aren't"
                )
            neighbours = np.rint(
                np.array(unformatted[2 : 2 + in_degree], dtype=float)
            ).astype(int)
            interaction = np.array(
                unformatted[2 + in_degree : 2 + 2 * in_degree], dtype=float
            )
            G.add_weighted_edges_from(
                list(zip(neighbours, [i] * in_degree, interaction))
            )  # neighbours are the genes pointing to i
        return G


def file_reader(filename):
    """Read the topology and regulatory files. Usage:
    input_regs = file_reader(input_file_regs)
    input_targets = file_reader(input_file_targets)
    """
    with open(filename) as f:
        data = []
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            data += [line[0].split(",")]
    return data


def run_sergio_1ct(input_regs, input_targets, number_sc=1000):
    """Run Sergio for one cell type"""
    n_MR, a = np.shape(input_regs)
    n_celltypes = a - 1
    id_cell_type = 0
    if n_celltypes != 1:
        warnings.warn(
            "Input containing "
            + str(n_celltypes)
            + " cell types, considering only cell type "
            + str(id_cell_type)
        )
    n_celltypes = 1
    input_regs = np.array(input_regs)[:, id_cell_type : 1 + id_cell_type + 1]
    n_reg_genes = len(input_targets)
    sim = sergio(
        number_genes=n_MR + n_reg_genes,
        number_bins=n_celltypes,
        number_sc=number_sc,
        noise_params=1,
        decays=0.8,
        sampling_state=15,
        noise_type="dpd",
    )
    sim.build_graph_array(
        input_taregts=input_targets, input_regs=input_regs, shared_coop_state=2
    )
    sim.simulate()
    expr = sim.getExpressions()
    expr_clean = np.concatenate(expr, axis=1)
    return expr_clean
