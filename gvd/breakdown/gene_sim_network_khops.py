import torch
import numpy as np
import networkx as nx
from functools import reduce
import operator
from tqdm.auto import tqdm # Import tqdm

class GeneSimNetworkKHops():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices
    """
    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """
        self.edge_list = edge_list
        self.gene_list = gene_list
        self.node_map = node_map
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        self._update_tensors()

    def _update_tensors(self):
        """Helper to regenerate tensors from the current nx.Graph state."""
        if not self.G.edges:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_weight = torch.empty((0,), dtype=torch.float)
            return

        # print(self.node_map)
        # print(self.G.edges)
        edge_index_ = [(self.node_map[e[0]], self.node_map[e[1]]) for e in self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T

        edge_attr = nx.get_edge_attributes(self.G, 'importance')
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

    # --- UPDATED METHOD ---

    def add_zero_weight_khop_edges(self, k, m):
        """
        For each node, finds all nodes reachable within k-hops. From this set,
        it calculates the max-multiplicative-strength path for each.
        It then adds 'm' zero-weight edges to the unconnected nodes with
        the highest strength.

        Args:
            k (int): The maximum hop distance to search.
            m (int): The number of new edges to add per node.
        """
        if k <= 0:
            print("k must be a positive integer.")
            return

        # Changed print statement
        print(f"Graph modification in progress. Original edge count: {len(self.edge_index[0])}")
        new_edges_to_add = []

        # Added tqdm wrapper to the main loop
        for start_node in tqdm(list(self.G.nodes()), desc="Processing nodes"):
            potential_edges = []

            # 1. OPTIMIZATION: Get the subgraph of all nodes reachable
            #    within k hops using nx.ego_graph.
            ego_graph = nx.ego_graph(self.G, n=start_node, radius=k)

            # 2. Iterate ONLY over this smaller set of reachable nodes
            for target_node in ego_graph.nodes():
                # Exclude self-loops and existing direct edges
                if start_node == target_node or self.G.has_edge(start_node, target_node):
                    continue

                # 3. Find all simple paths (this is the required slow part)
                #    We still search on the *original graph* (self.G)
                paths = nx.all_simple_paths(self.G,
                                            source=start_node,
                                            target=target_node,
                                            cutoff=k)

                max_strength = 0.0
                for path in paths:
                    if len(path) > 1:
                        # Calculate multiplicative strength
                        strength = reduce(operator.mul,
                                        (self.G[u][v]['importance'] for u, v in zip(path[:-1], path[1:])))
                        if strength > max_strength:
                            max_strength = strength

                # If a path was found, store this as a potential edge
                if max_strength > 0:
                    potential_edges.append((target_node, max_strength))

            # 4. Sort potential edges by their calculated strength
            potential_edges.sort(key=lambda x: x[1], reverse=True)

            # 5. Add the top 'm' new edges
            for target_node, _ in potential_edges[:m]:
                new_edges_to_add.append((start_node, target_node, {'importance': 0.0}))

        # 6. Add all new edges to the graph at once
        self.G.add_edges_from(new_edges_to_add)
        print(f"Added {len(new_edges_to_add)} new zero-weight edges.")

        # 7. Regenerate tensors to reflect the new graph structure
        self._update_tensors()
        print(f"Tensors updated. New edge count: {len(self.edge_index[0])}")