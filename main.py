import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
from matplotlib import cm


class RandomGraphAnalyzer:
    def __init__(self, num_nodes=50, num_edges=100):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.G = self.create_random_graph()
        self.node_levels = {}
        self.node_loads = {}
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

    def assign_random_levels(self):
        """Assign random levels to each node."""
        self.node_levels = {v: random.randint(0, 5) for v in self.G.nodes()}
        self.node_loads = {v: 0 for v in self.G.nodes()}

    def f(self, delta):
        """Define the function f(delta)."""
        return 1 / (1 + (1.1) ** (delta))

    def calculate_loads(self):
        """Calculate the loads based on node levels."""
        for u in self.G.nodes():
            self.node_loads[u] = 0
        for u, v in self.G.edges():
            delta = abs(self.node_levels[u] - self.node_levels[v])
            if self.node_levels[u] > self.node_levels[v]:
                if self.node_levels[v] == 0:
                    self.node_loads[v] += 1
                else:
                    self.node_loads[u] += self.f(delta)
                    self.node_loads[v] += 1 - self.f(delta)
            else:
                if self.node_levels[u] == 0:
                    self.node_loads[u] += 1
                else:
                    self.node_loads[v] += self.f(delta)
                    self.node_loads[u] += 1 - self.f(delta)

    def compute_median(self):
        """Compute the median load of nodes with load > 0."""
        positive_loads = [load for node, load in self.node_loads.items() if self.node_levels[node] > 0]
        return np.median(positive_loads) if positive_loads else 0

    def classify_nodes(self):
        """Classify nodes as up_dirty or down_dirty based on their loads."""
        median = self.compute_median()
        t = 1.1
        up_dirty = []
        down_dirty = []

        for node, load in self.node_loads.items():
            if load > t * median:
                up_dirty.append(node)
            elif load < median / (2*t) and self.node_levels[node] != 0:
                down_dirty.append(node)

        return up_dirty, down_dirty

    def adjust_levels(self):
        """Adjust levels of dirty nodes while printing the graph after each adjustment."""
        iteration = 0
        for i in range(20):
            up_dirty, down_dirty = self.classify_nodes()
            if not up_dirty and not down_dirty:
                break

            print(f"Iteration {iteration}: Adjusting levels")
            self.print_node_levels()
            print("up dirty:")
            print(up_dirty)
            print("down dirty:")
            print(down_dirty)

            # Adjust down_dirty nodes (lower levels)
            for node in down_dirty:
                if self.node_levels[node] > 0:  # Ensure level doesn't go below 0
                    self.node_levels[node] -= 1

            # Adjust up_dirty nodes (raise levels)
            for node in up_dirty:
                self.node_levels[node] += 1

            # Recalculate loads based on adjusted levels
            self.calculate_loads()

            # Normalize loads for coloring
            norm_loads = self.normalize_loads()
            colors = self.color_nodes_based_on_load(norm_loads)

            # Draw the graph with updated node colors based on new loads
            # if iteration % 10 == 0:
            self.draw_graph(f"Iteration {iteration}: Adjusted Graph", colors)

            iteration += 1
        print(f"Finished. Iteration: {iteration}")

    def normalize_loads(self):
        """Normalize the load values for coloring."""
        load_values = np.array(list(self.node_loads.values()))
        min_load = load_values.min()
        max_load = load_values.max()
        return (load_values) / (2 * max_load)

    def color_nodes_based_on_load(self, norm_loads):
        """Generate a grayscale colormap based on normalized load values."""
        return [cm.gray(1 - norm) for norm in norm_loads]  # Invert norm for color mapping

    def print_node_levels(self):
        """Print the levels and loads for all nodes."""
        print("Node Levels:")
        for node, level in self.node_levels.items():
            print(f"Node {node}: Level = {level}, Load = {self.node_loads[node]:.4f}")


if __name__ == '__main__':
    analyzer = RandomGraphAnalyzer()
    analyzer.assign_random_levels()
    analyzer.draw_graph("Random Graph with 20 Nodes and 50 Edges")

    analyzer.calculate_loads()
    norm_loads = analyzer.normalize_loads()
    colors = analyzer.color_nodes_based_on_load(norm_loads)

    analyzer.draw_graph("Graph with Node Colors Based on Load (Black = Max Load, White = Min Load)", colors)
    analyzer.print_node_levels()

    # Adjust levels of nodes until there are no dirty nodes
    analyzer.adjust_levels()

    # Print node levels and classifications
    analyzer.print_node_levels()
