import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class HyphalTracer:
    def __init__(self, adjacency_matrix, centroids):
        """
        Args:
            adjacency_matrix: NxN numpy array of connectivity scores (0.0 to 1.0).
            centroids: List of (row, col) tuples corresponding to the matrix indices.
        """
        # Store the adjacency matrix containing connectivity scores between all pairs of nuclei.
        self.matrix = adjacency_matrix
        # Store the list of (x, y) coordinates for each nucleus.
        self.centroids = centroids
        # Calculate the total number of nuclei (nodes) in the graph.
        self.num_nodes = len(centroids)
        
        # Nodes: the invidual nuclei
        # Parent: which node does the nuclei point to. ex. if self.parent[5]=2, it means node 5 points to node 2
        # Root: the representative of the connected nodes. A node is a root if it is its own parent, e.g. self.parent[2]==2
        # Initialize the Union-Find data structure.
        # self.parent is a list where index=node_id and value=parent_id.
        # Initially, every node is its own parent, meaning they are all in disjoint sets.
        self.parent = list(range(self.num_nodes))
        
    def _find(self, i):
            """Path compression to find the representative of the set."""
            # Check if node 'i' is its own parent. If yes, i is the root. If not, it's not the root.
            if self.parent[i] != i:
                # Recursively look at the parent node's parent, until we find the root node
                # Path Compression: Update self.parent[i] to point directly to the root.
                # This flattens the tree structure, making future lookups O(1) on average.
                self.parent[i] = self._find(self.parent[i])
            # Return the root representative of the set containing 'i'.
            return self.parent[i]

    def _union(self, i, j):
        """Unions the sets containing i and j."""
        # Find the root representative for node 'i'.
        root_i = self._find(i)
        # Find the root representative for node 'j'.
        root_j = self._find(j)
        
        # If the roots are different, the nodes are in different sets.
        if root_i != root_j:
            # Merge the sets by making the root of 'i' point to the root of 'j'.
            self.parent[root_i] = root_j
            # Return True to indicate a successful merge (connection made).
            return True
        # If roots are the same, they are already connected (cycle detected).
        return False

    def trace_hyphae(self, min_length=3, linearity_threshold=0.85, connectivity_threshold=0.0, angle_threshold_deg=90):
        """
        Reconstructs multiple hyphae from the connectivity matrix.
        
        Args:
            min_length: Minimum number of nuclei to consider a valid hypha.
            linearity_threshold: Ratio of (Euclidean Dist / Path Dist). 
                                 1.0 is a straight line. Lower means curved/zigzag.
            connectivity_threshold: Minimum score (0.0-1.0) to consider an edge valid.
            angle_threshold_deg: Minimum angle (degrees) allowed between edges at a node.
                                 Prevents sharp zig-zags. Default 90.
        
        Returns:
            valid_hyphae: List of dicts {'coords': [], 'indices': [], 'metrics': {}}
            rejected: List of dicts (same format) for chains that failed filters.
        """
        # 1. Create Edge List: (u, v, weight)
        edges = []
        removed_edges = 0
        
        # Get indices for the upper triangle of the matrix (excluding diagonal k=1).
        # This avoids duplicates (since matrix is symmetric) and self-loops.
        rows, cols = np.triu_indices_from(self.matrix, k=1)
        for r, c in zip(rows, cols):
            weight = self.matrix[r, c]
            # Change to >0 to keep only positive scores. Due to scaling, the lowest score will always be 0.
            if weight > 0: 
                if weight >= connectivity_threshold:
                    edges.append((r, c, weight))
                else:
                    removed_edges += 1
        
        print(f"Graph Construction: Kept {len(edges)} edges. Filtered {removed_edges} weak edges (< {connectivity_threshold}).")

        # 2. Sort Edges Descending (Global Greedy Strategy)
        # This is the "Greedy" part: we prioritize the highest scoring connections.
        # The lambda function tells sort to use the connectivity score for sorting.
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # 3. Build Graph with Constraints
        # Initialize a list to track the degree (number of connections) for each node.
        degrees = [0] * self.num_nodes
        # Track neighbors to calculate angles
        adj_list = {i: [] for i in range(self.num_nodes)}
        # List to store the edges that pass the constraints.
        selected_edges = []
        removed_angle = 0
        
        for u, v, w in edges:
            # CONSTRAINT 1: Degree <= 2 (Linear chain)
            # A nucleus in a linear hypha can have at most 2 neighbors.
            if degrees[u] < 2 and degrees[v] < 2:
                # CONSTRAINT 2: No Cycles (Acyclicity)
                # Check if u and v are already in the same set using Union-Find.
                if self._find(u) != self._find(v):
                    
                    # CONSTRAINT 3: Local Angle Check
                    # Ensure adding edge (u, v) doesn't create a sharp turn at u or v
                    def get_angle(center, n1, n2):
                        p_c = np.array(self.centroids[center])
                        p_1 = np.array(self.centroids[n1])
                        p_2 = np.array(self.centroids[n2])
                        
                        v1 = p_1 - p_c
                        v2 = p_2 - p_c
                        
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        
                        if norm1 == 0 or norm2 == 0: return 180.0
                        
                        dot = np.dot(v1, v2) / (norm1 * norm2)
                        dot = np.clip(dot, -1.0, 1.0)
                        return np.degrees(np.arccos(dot))

                    # Check angle at u if it already has a neighbor
                    if degrees[u] == 1:
                        neighbor = adj_list[u][0]
                        if get_angle(u, neighbor, v) < angle_threshold_deg:
                            removed_angle += 1
                            continue
                            
                    # Check angle at v if it already has a neighbor
                    if degrees[v] == 1:
                        neighbor = adj_list[v][0]
                        if get_angle(v, neighbor, u) < angle_threshold_deg:
                            removed_angle += 1
                            continue

                    # If disjoint, connect them (Union).
                    self._union(u, v)
                    # Increment degrees for both nodes.
                    degrees[u] += 1
                    degrees[v] += 1
                    adj_list[u].append(v)
                    adj_list[v].append(u)
                    # Add to the final list of edges.
                    selected_edges.append((u, v, w))
        
        print(f"Graph Construction: Filtered {removed_angle} edges due to sharp angles (< {angle_threshold_deg}°).")

        # 4. Analyze Components (The Forest)
        # Create a graph object to handle traversal logic.
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_weighted_edges_from(selected_edges)
        
        # Get all connected components (potential hyphae)
        components = list(nx.connected_components(G))
        
        # Some hyphae will be invalid due to length or linearity constraints
        valid_hyphae = []
        rejected = []
        
        for comp in components:
            # Skip single isolated nodes (noise)
            if len(comp) < 2:
                continue
                
            subgraph = G.subgraph(comp)
            
            # Find endpoints (nodes with degree 1)
            endpoints = [n for n, d in subgraph.degree() if d == 1]
            
            # Reconstruct the ordered path
            path_indices = []
            if len(endpoints) == 2:
                # Standard linear chain
                path_indices = nx.shortest_path(subgraph, source=endpoints[0], target=endpoints[1])
            elif len(endpoints) == 0 and len(comp) > 0:
                # Rare case: Perfect loop. Break arbitrarily.
                path_indices = list(comp)
            else:
                # Case: 2 nodes (both are endpoints)
                path_indices = list(comp)

            # --- Calculate Metrics ---
            
            # 1. Path Distance (Sum of physical distances between nuclei)
            path_dist = 0.0
            total_score = 0.0
            
            for k in range(len(path_indices) - 1):
                u, v = path_indices[k], path_indices[k+1]
                
                # Physical distance
                c1 = self.centroids[u]
                c2 = self.centroids[v]
                dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                path_dist += dist
                
                # Connectivity Score
                if G.has_edge(u, v):
                    total_score += G[u][v]['weight']
            
            avg_score = total_score / (len(path_indices) - 1) if len(path_indices) > 1 else 0
            
            # 2. Euclidean Distance (Crow flies distance start-to-end)
            start_c = self.centroids[path_indices[0]]
            end_c = self.centroids[path_indices[-1]]
            euclidean_dist = np.sqrt((start_c[0]-end_c[0])**2 + (start_c[1]-end_c[1])**2)
            
            # 3. Linearity (Tortuosity)
            # 1.0 = Straight line. < 0.5 = Very curvy/zigzag
            linearity = euclidean_dist / path_dist if path_dist > 0 else 0
            
            hypha_data = {
                'indices': path_indices,
                'coords': [self.centroids[i] for i in path_indices],
                'metrics': {
                    'length': len(path_indices),
                    'linearity': linearity,
                    'avg_score': avg_score
                }
            }
            
            # --- Apply Filters ---
            # We separate valid hyphae from "rejected" (debris/noise)
            # This ensures no nodes are silently "lost"
            if len(path_indices) >= min_length and linearity >= linearity_threshold:
                valid_hyphae.append(hypha_data)
            else:
                rejected.append(hypha_data)
                
        return valid_hyphae, rejected

def plot_hyphal_reconstruction(image, valid_hyphae, rejected_chains=None):
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    
    # 1. Plot Rejected Chains (Noise/Debris) - Faint Red
    if rejected_chains:
        for chain in rejected_chains:
            coords = chain['coords']
            if len(coords) > 1:
                ys, xs = zip(*coords)
                plt.plot(xs, ys, color='red', linewidth=2, alpha=0.25, linestyle='--')
    
    # 2. Plot Valid Hyphae - Distinct Colors
    # Generate distinct colors using the 'spring' colormap
    colors = plt.cm.spring(np.linspace(0, 1, len(valid_hyphae)))
    
    for idx, (hypha, color) in enumerate(zip(valid_hyphae, colors)):
        coords = hypha['coords']
        if len(coords) > 1:
            ys, xs = zip(*coords)
            
            # Plot the main path
            plt.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9)
            
            # Mark Start (Red) and End (Cyan) to show direction
            plt.plot(xs[0], ys[0], 'o', color='red', markersize=6, markeredgecolor='black', label='Start')
            plt.plot(xs[-1], ys[-1], 'o', color='cyan', markersize=6, markeredgecolor='black', label='End')
            
            # Label the hypha number at the midpoint
            mid = len(coords) // 2
            plt.text(xs[mid], ys[mid], str(idx+1), color='white', fontweight='bold', ha='center', va='center',
                     bbox=dict(facecolor=color, edgecolor='white', boxstyle='circle,pad=0.1', alpha=0.8))
    
    plt.title(f"Reconstruction: {len(valid_hyphae)} Hyphae, {len(rejected_chains) if rejected_chains else 0} Rejected")
    plt.legend()
    plt.axis('off')
    plt.show()

def calculate_hyphal_spacing_ratio(valid_hyphae, centroids, nuclei_lengths):
    """
    Calculates the ratio of internuclear distance to nuclei length.
    
    Ratio = (Average Distance between consecutive nuclei) / (Average Length of nuclei)
    
    Args:
        valid_hyphae: List of dicts, each containing 'indices' of the nuclei in order.
        centroids: List of (row, col) tuples.
        nuclei_lengths: List of major axis lengths corresponding to centroids.
        
    Returns:
        global_ratio: The ratio calculated using averages across all hyphae combined.
        per_hypha_ratios: List of ratios for each individual hypha.
    """
    all_distances = []
    all_lengths = []
    per_hypha_ratios = []
    
    for hypha in valid_hyphae:
        indices = hypha['indices']
        
        # Need at least 2 nuclei to have a distance
        if len(indices) < 2:
            continue
            
        # 1. Calculate distances between consecutive nuclei in this chain
        hypha_distances = []
        for k in range(len(indices) - 1):
            idx1 = indices[k]
            idx2 = indices[k+1]
            c1 = centroids[idx1]
            c2 = centroids[idx2]
            
            # Euclidean distance
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            hypha_distances.append(dist)
            all_distances.append(dist)
            
        # 2. Collect lengths of nuclei involved in this hypha
        hypha_lengths = [nuclei_lengths[i] for i in indices]
        all_lengths.extend(hypha_lengths)
        
        # 3. Calculate Ratio for this specific hypha
        avg_dist = np.mean(hypha_distances) if hypha_distances else 0
        avg_len = np.mean(hypha_lengths) if hypha_lengths else 0
        
        if avg_len > 0:
            ratio = avg_dist / avg_len
            per_hypha_ratios.append(ratio)
        else:
            per_hypha_ratios.append(0)
            
    # 4. Calculate Global Ratio (Weighted by number of nuclei/gaps)
    global_avg_dist = np.mean(all_distances) if all_distances else 0
    global_avg_len = np.mean(all_lengths) if all_lengths else 0
    
    if global_avg_len > 0:
        global_ratio = global_avg_dist / global_avg_len
    else:
        global_ratio = 0
        
    return global_ratio, per_hypha_ratios