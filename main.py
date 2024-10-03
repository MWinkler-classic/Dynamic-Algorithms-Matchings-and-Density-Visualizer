import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
from matplotlib import cm

class RandomGraphAnalyzer:
    def __init__(self, num_nodes=20, num_edges=20):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        # self.G = self.create_random_graph()
        self.node_levels = {}
        self.node_loads = {}
        self.pos = None
        self.G = self.initialize_city_on_a_hill_graph(40, 20, 10)
        # self.pos = nx.spring_layout(self.G)  # Layout is computed once to ensure consistent node positions

    def initialize_city_on_a_hill_graph(self, n, R, C):
        # Step 1: Generate a clique of size n/4
        clique_size = n // 4
        G = nx.complete_graph(clique_size)

        # Step 2: Add remaining nodes to the graph
        for i in range(clique_size, n):
            G.add_node(i)

        # Step 3: Add R random edges between non-clique nodes
        non_clique_nodes = list(range(clique_size, n))
        for _ in range(R):
            u, v = random.sample(non_clique_nodes, 2)  # Pick two different non-clique nodes
            G.add_edge(u, v)

        # Step 4: Add C edges connecting non-clique to clique nodes
        for _ in range(C):
            u = random.choice(non_clique_nodes)  # Pick a node from the non-clique
            v = random.choice(range(clique_size))  # Pick a node from the clique
            G.add_edge(u, v)

        # Step 5: Generate positions with shell layout
        self.pos = nx.shell_layout(G, [list(range(clique_size)), non_clique_nodes])

        return G

    def create_random_graph(self):
        """Create a random graph with specified number of nodes and edges."""
        return nx.gnm_random_graph(self.num_nodes, self.num_edges)

    def draw_graph(self, title, colors=None):
        """Draw the graph with highlighted matched edges."""
        plt.figure(figsize=(8, 6))

        # Separate matched and unmatched edges
        matched_edges = [(u, v) for u, v in self.G.edges() if self.G.nodes[u].get('partner') == v]
        unmatched_edges = [(u, v) for u, v in self.G.edges() if self.G.nodes[u].get('partner') != v]

        # Draw the unmatched edges
        nx.draw_networkx_edges(self.G, self.pos, edgelist=unmatched_edges, edge_color='gray')

        # Draw the matched edges in red
        nx.draw_networkx_edges(self.G, self.pos, edgelist=matched_edges, edge_color='red', width=2)

        # Draw the nodes
        nx.draw_networkx_nodes(self.G, self.pos, node_color=colors if colors else 'lightblue', node_size=500)
        nx.draw_networkx_labels(self.G, self.pos, font_size=10)

        # Add the title
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
            elif load < median / (2 * t) and self.node_levels[node] != 0:
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

    # Matching Algorithm Functions inside the Class
    def density(self, node):
        """Calculate the 'density' of a node as its load."""
        return self.node_loads[node]

    def sample_neighbors(self, v, C1):
        """Sample C1 neighbors with priority given to lower load (density) nodes."""
        neighbors = list(self.G.neighbors(v))
        densities = [(n, self.density(n)) for n in neighbors]

        # Sort neighbors by density (low to high)
        densities.sort(key=lambda x: x[1])

        sampled = []
        # Sample low-density neighbors first, but leave some slots for high-density neighbors
        for d in range(len(densities)):
            if len(sampled) >= C1:
                break
            sampled.append(densities[d][0])

        return sampled

    def match(self, v, u):
        """Match node v with node u."""
        self.G.nodes[v]['partner'] = u
        self.G.nodes[u]['partner'] = v

    def unmatch(self, v):
        """Unmatch a node from its partner."""
        partner = self.G.nodes[v].get('partner')
        if partner is not None:
            self.G.nodes[partner]['partner'] = None
        self.G.nodes[v]['partner'] = None

    def step(self, v, C1, C3, step_count, steps=0):
        """Recursive step of the algorithm with graph display."""
        if steps >= C3:
            return  # Termination condition

        sample = self.sample_neighbors(v, C1)

        # Check for unmatched neighbors
        norm_loads = self.normalize_loads()
        for neighbor in sample:
            if self.G.nodes[neighbor].get('partner') is None:
                self.match(v, neighbor)
                # Normalize loads for coloring
                colors = self.color_nodes_based_on_load(norm_loads)
                self.draw_graph(f"Matched {v} with {neighbor} (Step {steps})", colors)
                return  # Exit after matching

        # Find the neighbor with the highest load partner
        max_load = -1
        v_prime = None
        M = None

        for neighbor in sample:
            partner = self.G.nodes[neighbor].get('partner')
            if partner is not None:
                partner_load = self.density(partner)
                if partner_load > max_load:
                    max_load = partner_load
                    v_prime = neighbor
                    M = partner

        if v_prime is not None and M is not None:
            # Unmatch v_prime and M
            self.unmatch(v_prime)

            # Match v with v_prime
            self.match(v, v_prime)

            # Normalize loads for coloring
            norm_loads = self.normalize_loads()
            colors = self.color_nodes_based_on_load(norm_loads)

            # Display graph after matching
            self.draw_graph(f"Unmatched {v_prime} from {M}, Matched {v} with {v_prime} (Step {steps})", colors)

            # Call step(M)
            self.step(M, C1, C3, step_count, steps + 1)


    def run_matching_algorithm(self, C1=30, C3=50):
        """Run the matching algorithm on the graph."""
        for node in self.G.nodes:
            self.G.nodes[node]['partner'] = None  # Initialize partner as None

        nodes_by_load = sorted(self.G.nodes, key=lambda x: self.node_loads[x])

        for u in nodes_by_load:
            if self.G.nodes[u].get('partner') is None:  # Only run for unmatched nodes
                self.step(u, C1, C3, step_count=0)
        for u in nodes_by_load:
            if self.G.nodes[u].get('partner') is None:  # Only run for unmatched nodes
                self.step(u, C1, C3, step_count=0)


if __name__ == '__main__':
    analyzer = RandomGraphAnalyzer()
    analyzer.assign_random_levels()
    analyzer.draw_graph("Initial Random Graph")

    analyzer.calculate_loads()
    analyzer.adjust_levels()
    analyzer.print_node_levels()

    # Run matching algorithm after load balancing
    analyzer.run_matching_algorithm(C1=30, C3=50)

    analyzer.draw_graph("Final Graph with Matching")
