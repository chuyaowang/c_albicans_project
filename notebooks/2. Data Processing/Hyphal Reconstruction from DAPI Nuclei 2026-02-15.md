# Hyphal Reconstruction from DAPI Nuclei

#candida 

## DAPI artifacts

Sometimes DAPI staining shows things that do not look like nuclei

- mammalian nuclei out of focus
- C. albicans nuclei out of focus
- mitochondrial DNA
- contamination, left-over of DAPI dye

---
> Algorithm Documentation: Automated Hyphal Reconstruction

## 1. Overview

The goal of this algorithm is to automatically identify fungal nuclei within a microscopy image and reconstruct the linear structure of the hypha (the elongated cell) by connecting these nuclei in the correct biological order.

The process is divided into four main phases:

1. **Segmentation:** Identifying where the nuclei are.
2. **Filtering:** Removing noise and artifacts.
3. **Connectivity Analysis:** Scoring how likely two nuclei are to be neighbors.
4. **Reconstruction:** Tracing the single "best" path through the nuclei.

Also, prior to the algorithm, the input image is pre-processed using filters to remove artifacts and enhance contrast.

---

## Phase 1: Image Segmentation (Finding the Nuclei)

Before we can connect nuclei, we must locate them. We use two different image inputs for this process to maximize accuracy:

- **Segmentation Source:** A pre-processed image (often blurred or contrast-enhanced) used solely to detect shapes.
- **Intensity Source:** The original raw image, used to measure brightness values.

### Step 1.1: Thresholding

We use **Otsu’s Method** to automatically separate the foreground (nuclei) from the background. Otsu’s method calculates a specific brightness threshold that minimizes the variance within the two classes (black background vs. white nuclei), creating a binary "mask" (black and white image).

### Step 1.2: Watershed Separation (Optional)

In fungal hyphae, nuclei are often close together and may appear to touch in the binary mask, looking like a single peanut shape rather than two distinct circles.

- **The Problem:** If we treat touching nuclei as one object, our count will be wrong.
- **The Solution:** We use the **Watershed Algorithm**. Imagine the brightness of the image as a topographic map where nuclei are mountain peaks. We "flood" the landscape with water. Where the water from two different peaks meets, we build a dam (a dividing line). This effectively splits touching nuclei into distinct objects.

---

## Phase 2: Filtering (Quality Control)

Not every bright spot in an image is a nucleus. It could be dust, noise, or a staining artifact.

### Step 2.1: Size Filtering

We calculate the area (in pixels) of every detected object.

- We compute the **Mean Area** of all objects.
- We discard objects that are significantly smaller (e.g., <33% of the mean) or significantly larger (e.g., >300% of the mean) than the average.
- **Result:** A clean list of "Valid Centroids" (x, y coordinates) representing the center of each true nucleus.

---

## Phase 3: Connectivity Analysis (Scoring Connections)

Now that we have a set of points, we need to determine which ones are neighbors. We cannot simply connect the closest points, because hyphae can curve, and a nucleus might be physically close to another but separated by a cell wall or gap.

We calculate a **Connectivity Score** for every possible pair of nuclei based on two factors:

### Factor A: The "Signal" (Intensity)

We draw a straight line between two nuclei and measure the brightness of the pixels along that path.

- **Refinement:** We mask out the nuclei themselves. We only care about the "bridge" (cytoplasm) _between_ them, not the brightness of the nuclei.
- **Logic:** A bright path suggests a continuous cytoplasmic connection (part of the same hypha). A dark path suggests empty space or background.

### Factor B: The "Cost" (Distance)

We measure the physical Euclidean distance between the two nuclei.

- **Logic:** Biologically, neighboring nuclei in a hypha are usually spaced somewhat regularly. Extremely long connections are unlikely.

### Factor C: The "Alignment" (Orientation)

Nuclei are typically ellipsoidal, and their major axis usually aligns with the hyphal growth direction.

- **Logic:** We calculate the angle between the longest axis of the nucleus and the vector connecting it to the neighbor. There are two angles in each pair of nuclei, we use the minimum of the two.
- **Penalty:** We apply a penalty based on the cosine of this angle. A connection perpendicular to the nucleus orientation is heavily penalized, while a parallel connection retains its full score. This penalty modifies the **Intensity** score.

### Step 3.1: Normalization (Balancing the Scales)

Pixel intensity (0–255) and Distance (0–1000+ pixels) are "apples and oranges." To combine them, we perform **Min-Max Normalization**:

1. We find the brightest and dimmest connections in the entire image and scale all Intensity scores to a **0.0 to 1.0** range.
2. We find the longest and shortest distances and scale all Distance scores to a **0.0 to 1.0** range.

### Step 3.2: The Final Formula

$$ \text{Score} = (\text{Normalized Average Intensity} - (\alpha \times \text{Normalized Distance}))\times \cos(Alignment Angle) $$

A high score means the connection is **bright** and **short**. A low score means the connection is dark or very far away.

---

## Phase 4: Hyphal Reconstruction (The Tracer)

This is the most complex step. We have a "web" of possible connections (an adjacency matrix), where every nucleus is theoretically connected to every other nucleus with a specific score.

We need to trim this web down to a single, linear line (the hypha) that passes through the nuclei in order. We use a **Greedy Edge Sorting Algorithm** (a variation of Kruskal's Algorithm).

### The Logic of Reconstruction:

Imagine you have a pile of potential "sticks" (connections) to build a track. Some sticks are strong (high score), some are weak (low score).

1. **Ranking:** We list every possible connection and sort them from **highest score to lowest score**. We want to prioritize the strongest, most obvious connections first.
    
2. **Selection Loop:** We pick up the "best" stick (highest score) and ask two questions before adding it to our hypha:
    
    - **Constraint 1: The "Two-Hand" Rule (Degree Constraint)** A nucleus in a linear hypha can only have two neighbors: one "upstream" and one "downstream."
        
        - _Check:_ Does Nucleus A already have 2 connections? Does Nucleus B already have 2 connections?
        - _Action:_ If either is "full," we throw this stick away. We cannot branch.
    - **Constraint 2: The "No-Loop" Rule (Acyclicity)** Hyphae are linear; they do not loop back on themselves to form a circle.
        
        - _Check:_ Are Nucleus A and Nucleus B already part of the same connected chain?
        - _Action:_ If they are already connected (indirectly) via other nodes, adding this connection would close a loop. We throw this stick away.
3. **Merging:** If a connection passes both rules, we add it. This might connect two isolated nuclei, or it might join two short chains into one longer chain.
    
4. **Final Path:** After checking all possible connections, we are left with the mathematically optimal linear path that maximizes the signal strength along the hypha. We then traverse this path from one endpoint to the other to get the ordered list of coordinates

> [!info] Union-Find Data Structure
> The **Union-Find** (also known as **Disjoint Set Union** or **DSU**) is a highly efficient data structure used to track elements that are split into a number of disjoint (non-overlapping) sets.
> 
> It is primarily used to answer the question: _"Are these two items already connected?"_ and to perform the action: _"Connect these two items."_
> 
> ### 1. The Concept: "Islands and Bridges"
> 
> Imagine every nucleus in your image starts as its own isolated island.
> 
> - **Find:** This operation asks, "Which island cluster does this nucleus belong to?" It returns a representative ID (like a "leader") for that cluster.
> - **Union:** This operation builds a bridge between two islands, effectively merging them into one larger continent.
> 
> ### 2. What It Does In Your Algorithm
> 
> In the context of your **Hyphal Reconstruction (Greedy Edge Sorting)**, the Union-Find structure serves one critical purpose: **Cycle Detection (preventing loops).**
> 
> Biological hyphae are linear structures; they grow in lines or branches, but they almost never loop back on themselves to form a closed circle (like a donut).
> 
> Here is exactly what happens in your `trace_longest_path` function:
> 
> 1. **Initialization:** At the start, `self.parent = [0, 1, 2, ...]` means every nucleus is its own parent. Nucleus 0 is in Set 0, Nucleus 1 is in Set 1. They are all disconnected.
>     
> 2. **The Loop (Processing Edges):** You iterate through your connections from strongest (highest score) to weakest. Let's say you are looking at a connection between **Nucleus A** and **Nucleus B**.
>     
> 3. **The "Find" Check:** The code calls `self._find(A)` and `self._find(B)`.
>     
>     - **Scenario 1: Different Sets** If `Find(A)` returns 5 and `Find(B)` returns 8, it means A and B are currently in different chains.
>         - **Action:** It is safe to connect them! You call `_union(A, B)`. Now, anyone connected to A is also connected to B.
>     - **Scenario 2: Same Set (The Trap)** If `Find(A)` returns 5 and `Find(B)` _also_ returns 5, it means A and B are **already connected** indirectly (perhaps A is connected to C, and C is connected to B).
>         - **Action:** If you add a direct link between A and B now, you will create a closed loop (A -> C -> B -> A).
>         - **Result:** The algorithm **skips** this connection to preserve the linear structure of the hypha.
> 
> ### 3. Why use this instead of a simple list?
> 
> You might wonder, _"Why not just keep a list of connected nodes?"_
> 
> The magic of Union-Find is **speed**. If you have a long chain of 100 nuclei, and you want to check if Nucleus #1 is connected to Nucleus #100, a standard list search would be slow ($O(N)$).
> 
> Union-Find uses a trick called **Path Compression** (seen in your code: `self.parent[i] = self._find(self.parent[i])`). This flattens the internal tree structure, making the `Find` operation nearly instantaneous ($O(1)$ on average). This allows your algorithm to sort and check thousands of potential connections in a fraction of a second.
> 
> ### Summary
> 
> - **Input:** Two nuclei you want to connect.
> - **Union-Find asks:** "Are they already in the same family?"
> - **If No:** Connect them (Union).
> - **If Yes:** Don't connect them (avoids loops).

---

## 5. Implementation: The HyphalTracer Class

The following Python class implements the logic described above. It takes the adjacency matrix of connectivity scores and the list of nucleus centroids as input. It uses the Union-Find data structure to prevent cycles and a greedy strategy to build the longest linear chain.

```python
class HyphalTracer:
    def __init__(self, adjacency_matrix, centroids):
        """
        Args:
            adjacency_matrix: NxN numpy array of connectivity scores (0.0 to 1.0).
            centroids: List of (row, col) tuples corresponding to the matrix indices.
        """
        # Store the adjacency matrix containing scores for all node pairs
        self.matrix = adjacency_matrix
        # Store the list of coordinates (centroids) for the nuclei
        self.centroids = centroids
        # Count the total number of nodes (nuclei)
        self.num_nodes = len(centroids)
        
        # Initialize Union-Find parent array. 
        # Each node starts as its own parent (disjoint sets).
        # This is used to detect cycles during reconstruction.
        self.parent = list(range(self.num_nodes))
        
    def _find(self, i):
        """Path compression to find the representative of the set."""
        # If i is not its own parent, it is not the root
        if self.parent[i] != i:
            # Recursively find the root and update the parent pointer (Path Compression)
            # This flattens the tree, making future operations O(1)
            self.parent[i] = self._find(self.parent[i])
        # Return the root of the set
        return self.parent[i]

    def _union(self, i, j):
        """Unions the sets containing i and j."""
        # Find the root of the set containing i
        root_i = self._find(i)
        # Find the root of the set containing j
        root_j = self._find(j)
        
        # If they have different roots, they are in different sets
        if root_i != root_j:
            # Merge the sets by making one root point to the other
            self.parent[root_i] = root_j
            # Return True indicating a merge occurred
            return True
        # If roots are the same, they are already connected (cycle detected)
        return False

    def trace_longest_path(self):
        """
        Executes Greedy Edge Sorting with Degree Constraints.
        Returns:
            ordered_centroids: List of (row, col) tuples in order from one end to the other.
            path_indices: List of indices corresponding to the centroids.
        """
        # 1. Create Edge List: (u, v, weight)
        edges = []
        # Get indices for the upper triangle of the matrix (excluding diagonal)
        # We only need one direction (u, v) since the matrix is symmetric
        rows, cols = np.triu_indices_from(self.matrix, k=1)
        for r, c in zip(rows, cols):
            weight = self.matrix[r, c]
            # Optimization: Only consider edges with positive connectivity scores
            if weight > 0: 
                edges.append((r, c, weight))
        
        # 2. Sort Edges Descending (Strongest connections first)
        # This implements the "Greedy" strategy: prioritize the best links
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # 3. Greedy Selection
        # Track the degree (number of connections) for each node
        degrees = [0] * self.num_nodes
        # List to store the edges accepted into the final graph
        selected_edges = []
        
        for u, v, w in edges:
            # CONSTRAINT 1: Degree <= 2 (Linear chain)
            # A nucleus in a hypha can only have 2 neighbors (upstream/downstream)
            if degrees[u] < 2 and degrees[v] < 2:
                # CONSTRAINT 2: No Cycles (Acyclicity)
                # Check if u and v are already connected indirectly
                if self._find(u) != self._find(v):
                    # If not connected, union them
                    self._union(u, v)
                    # Update degrees
                    degrees[u] += 1
                    degrees[v] += 1
                    # Add to selected edges
                    selected_edges.append((u, v))
        
        # 4. Reconstruct Path using NetworkX
        # Build a graph from the selected edges to analyze the structure
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(selected_edges)
        
        # Get all connected components (separate hyphae chains)
        components = list(nx.connected_components(G))
        
        if not components:
            return [], []

        # Find the largest component (assuming it is the main hypha)
        largest_comp = max(components, key=len)
        subgraph = G.subgraph(largest_comp)
        
        # Find endpoints: Nodes with degree 1 in the chain
        endpoints = [n for n, d in subgraph.degree() if d == 1]
        
        path_indices = []
        
        if not endpoints:
            # Case: Single node or isolated nodes
            if len(largest_comp) == 1:
                path_indices = list(largest_comp)
            else:
                # Rare Case: A perfect loop (should be prevented by Union-Find)
                path_indices = list(largest_comp)
        else:
            # Traverse from one endpoint to the other
            start_node = endpoints[0]
            end_node = endpoints[1]
            # Find the shortest path (which is the only path in a tree/line)
            path_indices = nx.shortest_path(subgraph, source=start_node, target=end_node)
            
        # Map indices back to (row, col) coordinates
        ordered_centroids = [self.centroids[i] for i in path_indices]
        
        return ordered_centroids, path_indices
```
