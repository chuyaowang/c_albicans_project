import numpy as np
import networkx as nx


def geodesic_distance_map(mask, tip_coord, pixel_size_um=1.0):
    """Compute the geodesic (along-mask) distance from a tip pixel to every foreground pixel.

    Builds an 8-connected pixel graph restricted to foreground pixels and runs
    Dijkstra's algorithm from *tip_coord*.  Edge weights are Euclidean pixel
    distances (1.0 for axis-aligned, √2 for diagonal), scaled by *pixel_size_um*.

    Parameters
    ----------
    mask : ndarray (bool or int)
        2-D binary mask of the cell/hypha.
    tip_coord : (y, x) tuple
        Source pixel for the distance computation (must be inside the mask).
    pixel_size_um : float
        Conversion factor from pixels to micrometres.

    Returns
    -------
    ndarray (float)
        Array of the same shape as *mask*; background pixels are 0, foreground
        pixels contain their geodesic distance from *tip_coord* in µm.
    """
    rows, cols = mask.shape
    G = nx.Graph()
    for y in range(rows):
        for x in range(cols):
            if mask[y, x]:
                G.add_node((y, x))
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx_ < cols:
                            if mask[ny, nx_]:
                                G.add_edge(
                                    (y, x), (ny, nx_), weight=np.hypot(dy, dx)
                                )
    length_dict = nx.single_source_dijkstra_path_length(
        G, source=tip_coord, weight="weight"
    )
    dist_map = np.zeros_like(mask, dtype=float)
    for (y, x), dist in length_dict.items():
        dist_map[y, x] = dist * pixel_size_um
    return dist_map