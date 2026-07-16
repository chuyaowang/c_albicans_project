import numpy as np
import networkx as nx


def build_skeleton_graph(skeleton):
    """Convert a binary skeleton image into an undirected NetworkX graph.

    Each foreground pixel is a node; 8-connected neighbours share an edge.
    Nodes with exactly one neighbour are endpoints; nodes with 3+ are branch points.
    """
    G = nx.Graph()
    rows, cols = skeleton.shape
    for y in range(rows):
        for x in range(cols):
            if skeleton[y, x]:
                G.add_node((y, x))
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx_ < cols:
                            if skeleton[ny, nx_]:
                                G.add_edge((y, x), (ny, nx_))
    return G


def extend_skeleton(skel, endpoint, max_steps=20):
    """Linearly extrapolate a skeleton outward from an endpoint.

    skimage.medial_axis tends to stop short of the true cell boundary at tips.
    This function projects the skeleton in the direction of the last segment
    for up to *max_steps* pixels, stopping when it would leave the image or
    re-enter the existing skeleton.

    Parameters
    ----------
    skel : ndarray (int or bool)
        Binary skeleton image.
    endpoint : (y, x) tuple
        The skeleton pixel to extend from (must have exactly one neighbour).
    max_steps : int
        Maximum number of pixels to add.

    Returns
    -------
    ndarray
        A copy of *skel* with the extension applied.
    """
    y, x = endpoint
    skel_extended = skel.copy()
    neighbors = [
        (y + dy, x + dx)
        for dy in [-1, 0, 1]
        for dx in [-1, 0, 1]
        if (dy != 0 or dx != 0)
        and 0 <= y + dy < skel.shape[0]
        and 0 <= x + dx < skel.shape[1]
        and skel[y + dy, x + dx]
    ]
    if not neighbors:
        return skel_extended
    ny, nx_ = neighbors[0]
    dy, dx = y - ny, x - nx_
    for _ in range(max_steps):
        y_new, x_new = int(round(y + dy)), int(round(x + dx))
        if not (0 <= y_new < skel.shape[0] and 0 <= x_new < skel.shape[1]):
            break
        if skel_extended[y_new, x_new] == 0:
            skel_extended[y_new, x_new] = 1
            y, x = y_new, x_new
        else:
            break
    return skel_extended