import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from collections import defaultdict

# TODO: fix some nodes getting negative load
# fix raising a level does not lower node properly
# paste code to chatGPT for debugging

# constant constants
ACTIVE = 0
INACTIVE = 1

# Algorithmic constants
EPSILON = 0.05
d = 4  # The divisor for level difference modulo d
D = 20  # Threshold or parameter for adjustments

# run constants
NODES = 200
EDGES = 2000
MAX_ITERATIONS = 1000
PRINT_INTERVAL = MAX_ITERATIONS/10


class RandomGraphAnalyzer:
    # -----------------------------INIT FUNCTIONS-------------------------------------#
    def __init__(self, num_nodes=NODES, num_edges=EDGES):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.G = self.create_random_graph()
        self.node_levels = {}
        self.node_loads = {}
        self.remainder_lists = {}  # To store neighbors grouped by level difference mod d
        self.last_seen_index = defaultdict(lambda: defaultdict(dict))
        self.remainders_index = {}  # To store the last index checked by each node
        self.pos = nx.spring_layout(self.G)  # Layout is computed once to ensure consistent node positions

    def create_random_graph(self):
        """Create a random graph with specified number of nodes and edges."""
        return nx.gnm_random_graph(self.num_nodes, self.num_edges)

    def assign_starting_levels(self):
        """Assign random levels to each node."""
        self.node_levels = {v: 0 for v in self.G.nodes()}
        # self.node_loads = {v: 0 for v in self.G.nodes()}
        # Initialize remainder_lists and last_checked_index
        for node in self.G.nodes():
            self.remainders_index[node] = 0  # Start with index 0

            self.remainder_lists[node] = {i: [[], []] for i in range(d)}
            # every remainder has an active list(index 0) and a passive list(index 1)
            for nbr in self.G.neighbors(node):
                self.last_seen_index[node][D][nbr] = (nbr, 0, 0, 0.5)
            # [[[],[]] for nbr in self.G.neighbors(node)]

    def draw_graph(self, title, colors=None):
        """Draw the graph with given title and node colors, using consistent positions."""
        plt.figure(figsize=(8, 6))
        nx.draw(self.G, self.pos, with_labels=True, node_color=colors if colors else 'lightblue',
                edge_color='gray', node_size=500, font_size=10)
        plt.title(title)
        plt.show()

    def calculate_starting_loads(self):
        """Calculate the loads based on node levels."""
        for u in self.G.nodes():
            self.node_loads[u] = self.G.degree(u) / 2  # every edge u has gives half its load to u.

    # -----------------------------GRAPHICAL FUNCTIONS--------------------------------#

    def normalize_loads(self):
        """Normalize the load values for coloring."""
        load_values = np.array(list(self.node_loads.values()))
        min_load = load_values.min()
        max_load = load_values.max()
        return (load_values - min_load) / (max_load - min_load)  # Normalize to [0, 1]

    def color_nodes_based_on_load(self, norm_loads):
        """Generate a grayscale colormap based on normalized load values."""
        return [cm.gray(1 - norm) for norm in norm_loads]  # Invert norm for color mapping

    def print_node_levels(self):
        """Print the levels and loads for all nodes."""
        print("Node Levels:")
        for node, level in self.node_levels.items():
            print(f"Node {node}: Level = {level}, Load = {self.node_loads[node]:.4f}")

    # -----------------------------ALGORTITHM FUNCTIONS-------------------------------#

    def f(self, delta):
        return max(0.0, min(1.0, EPSILON * round((D - delta) / d)))

    def classify_nodes(self):
        """Classify nodes as up_dirty based on their loads."""
        max_load = max(self.node_loads.values())
        t = EPSILON
        up_dirty = []

        print("max load = " + str(max_load) + "\n")
        for node, load in self.node_loads.items():
            if load > max_load - EPSILON:
                up_dirty.append(node)

        return up_dirty

    def adjust_levels(self):
        """Adjust levels of dirty nodes while printing the graph after each adjustment."""
        iteration = 0
        for i in range(MAX_ITERATIONS):
            up_dirty = self.classify_nodes()
            if not up_dirty:
                break

            print(f"Iteration {iteration}: Adjusting levels")
            self.print_node_levels()
            print("up dirty:")
            print(up_dirty)

            # Adjust up_dirty nodes (raise levels)
            for node in up_dirty:
                self.raise_level(node)

            # Normalize loads for coloring
            norm_loads = self.normalize_loads()
            colors = self.color_nodes_based_on_load(norm_loads)

            # Draw the graph with updated node colors based on new loads
            if iteration % PRINT_INTERVAL == 0:
                self.draw_graph(f"Iteration {iteration}: Adjusted Graph", colors)

            iteration += 1
        print(f"Finished. Iteration: {iteration}")

    def raise_level(self, node):
        self.node_levels[node] += 1

        # check the neighbors in the remainder list corresponding to the old index
        current_index = self.remainders_index[node]
        while self.remainder_lists[node][current_index][ACTIVE]:
            nbr, last_node_level, last_nbr_level, last_load_on_node  = self.remainder_lists[node][current_index][ACTIVE].pop()
            self.check_nbr(node, nbr, last_node_level, last_nbr_level, last_load_on_node)

        self.remainder_lists[node][current_index][ACTIVE] = self.remainder_lists[node][current_index][INACTIVE]
        self.remainder_lists[node][current_index][INACTIVE] = []

        # Create a copy of items to iterate over to avoid modifying the dictionary during iteration
        items_to_check = list(self.last_seen_index[node][self.node_levels[node]].items())
        for key, value in items_to_check:
            nbr, last_node_level, last_nbr_level, last_load_on_node = value
            self.check_nbr(node, nbr, last_node_level, last_nbr_level, last_load_on_node)

        self.last_seen_index[node][self.node_levels[node]] = {}

        # After raising the level, increment the last checked index for this node
        self.remainders_index[node] = (self.remainders_index[node] + 1) % d

    def check_nbr(self, node, nbr, last_node_level, last_nbr_level, last_load_on_node):
        last_seen_delta = last_node_level - last_nbr_level
        delta = self.node_levels[node] - self.node_levels[nbr]

        # redistribute the load
        if delta < 0:
            return  # nbr is now higher than node
        elif delta > last_seen_delta + d / 2:
            # remove previous loads
            self.node_loads[node] -= last_load_on_node
            self.node_loads[nbr] -= 1.0 - last_load_on_node
            # add new loads
            new_load_on_node = self.f(delta)
            self.node_loads[node] += new_load_on_node
            self.node_loads[nbr] += 1.0 - new_load_on_node
        else:
            new_load_on_node = last_load_on_node

        # insert into correct indexes at lists
        if node in self.last_seen_index[nbr][last_node_level]:
            del self.last_seen_index[nbr][last_node_level][node]  # remove node from nbrs list
        if nbr in self.last_seen_index[node][last_nbr_level]:
            del self.last_seen_index[node][last_nbr_level][nbr]  # remove nbr from node list

        self.last_seen_index[nbr][self.node_levels[node]][node] = (node, self.node_levels[nbr],
                                                                       self.node_levels[node], new_load_on_node)  # add node to nbrs list
        if delta < D + d / EPSILON:
            if delta >= D:
                # nbr is in the range where load is changing, D + d/epsilon > delta > D
                self.remainder_lists[node][(delta-D) % d][INACTIVE].append((nbr, self.node_levels[node],
                                                                       self.node_levels[nbr], new_load_on_node))
            else:
                # nbr is below node but above changing range, D > delta > 0
                self.last_seen_index[node][self.node_levels[nbr]][nbr] = (nbr, self.node_levels[node],
                                                                       self.node_levels[nbr], new_load_on_node)





if __name__ == '__main__':
    analyzer = RandomGraphAnalyzer()
    analyzer.assign_starting_levels()
    analyzer.draw_graph("Random Graph")

    analyzer.calculate_starting_loads()

    # Adjust levels of nodes until there are no dirty nodes
    analyzer.adjust_levels()

    # # Print node levels and classifications
    # analyzer.print_node_levels()
