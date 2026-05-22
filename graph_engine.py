import networkx as nx
import torch

class EvidenceGraphManager:
    """
    Core engine for constructing heterogeneous evidence graphs (Section 3.2).
    Manages structural dependencies and multi-hop retrieval space.
    """
    def __init__(self, config):
        self.config = config
        self.graph = nx.DiGraph()

    def update_edge_weights(self, u, v, s_sem, s_sup, s_cit, s_con):
        """Eq (1): w(i,j) calculation for the evidence graph."""
        weight = (self.config['alpha_w'] * s_sem + 
                  self.config['beta_w'] * s_sup + 
                  self.config['gamma_w'] * s_cit - 
                  self.config['delta_w'] * s_con)
        self.graph.add_edge(u, v, weight=weight)
        return weight
