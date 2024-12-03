import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Constants
MAX_DELTA = 10
L_TAG_RANGE = 1
EPSILON = 0.05
d = 5  # The divisor for level difference modulo d
D = 2  # Threshold or parameter for adjustments


class RandomGraphAnalyzer:
    def __init__(self, num_nodes=40, num_edges=200):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.G = self.create_random_graph()
        self.node_levels = {}
        self.node_loads = {}
        self.remainder_lists = {}  # To store neighbors grouped by level difference mod d
        self.last_checked_index = {}  # To store the last index checked by each node
        self.pos = nx.spring_layout(self.G)  # Layout is computed once to ensure consistent node positions

    def create_random_graph(self):
        """Create a random graph with specified number of nodes and edges."""
        return nx.gnm_random_graph(self.num_nodes, self.num_edges)

    def draw_graph(self, title, colors=None):
        """Draw the graph with given title and node colors, using consistent positions."""
        plt.figure(figsize=(8, 6))
        nx.draw(self.G, self.pos, with_labels=True, node_color=colors if colors else 'lightblue',
                edge_color='gray', node_size=500, font_size=10)
        plt.title(title)
        plt.show()

    def assign_starting_levels(self):
        """Assign random levels to each node."""
        self.node_levels = {v: 0 for v in self.G.nodes()}
        self.node_loads = {v: 0 for v in self.G.nodes()}
        # Initialize remainder_lists and last_checked_index
        for node in self.G.nodes():
            self.remainder_lists[node] = {i: [] for i in range(d)}  # Initialize lists for each remainder
            self.last_checked_index[node] = 0  # Start with index 0

    def compute_max_load(self):
        return max(self.node_levels.values())

    def calculate_starting_loads(self):
        """Calculate the loads based on node levels."""
        for u in self.G.nodes():
            self.node_loads[u] = self.G.degree(u) / 2  # every edge u has gives half its load to u.

    def classify_nodes(self):
        """Classify nodes as up_dirty or down_dirty based on their loads."""
        max_load = self.compute_max_load()
        t = EPSILON
        up_dirty = []

        print("max load = " + str(max_load) + "\n")
        for node, load in self.node_loads.items():
            if load > t * max_load - EPSILON:
                up_dirty.append(node)

        return up_dirty

    def adjust_levels(self):
        """Adjust levels of dirty nodes while printing the graph after each adjustment."""
        iteration = 0
        for i in range(200):
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
            if iteration % 10 == 0:
                self.draw_graph(f"Iteration {iteration}: Adjusted Graph", colors)

            iteration += 1
        print(f"Finished. Iteration: {iteration}")

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

    def raise_level(self, node):
        self.node_levels[node] += 1

        # Update remainder_lists for neighbors based on the new level difference
        for neighbor in self.G.neighbors(node):
            level_diff = (self.node_levels[node] - self.node_levels[neighbor]) % d
            self.remainder_lists[node][level_diff].append(neighbor)
            self.remainder_lists[neighbor][level_diff].append(node)

        # After raising the level, increment the last checked index for this node
        self.last_checked_index[node] = (self.last_checked_index[node] + 1) % d

        # Now, check the neighbors in the remainder list corresponding to the new index
        current_index = self.last_checked_index[node]
        for neighbor in self.remainder_lists[node][current_index]:
            self.node_loads[neighbor] += 1  # Example of how load could propagate (this is simplified)


if __name__ == '__main__':
    analyzer = RandomGraphAnalyzer()
    analyzer.assign_starting_levels()
    analyzer.draw_graph("Random Graph")

    analyzer.calculate_starting_loads()

    # Adjust levels of nodes until there are no dirty nodes
    analyzer.adjust_levels()

    # Print node levels and classifications
    analyzer.print_node_levels()
