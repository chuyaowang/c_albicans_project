import numpy as np
import pandas as pd
from skimage.measure import profile_line
import matplotlib.pyplot as plt

def extract_graph(valid_nuclei_data, int_image, binary_mask_filled, path_width=3, show_plot=True, show_edge=False):
    """
    Extracts features for nuclei and paths as pandas DataFrames, and the edge indices.

    Args:
        valid_nuclei_data (list): List of properties for valid nuclei.
        int_image (numpy.ndarray): The intensity image used to measure signal between nuclei.
        binary_mask_filled (numpy.ndarray): Mask used to exclude the nuclei pixels themselves.
        path_width (int): Width of the line profile used to sample intensity between centroids.
        show_plot (bool): Whether to display a plot of the extracted graph over the image.

    Returns:
        tuple: A tuple containing:
            - nuclei_df (pandas.DataFrame): Features for each nucleus.
            - paths_df (pandas.DataFrame): Features for each path between nuclei.
            - edge_index (list of lists): A list of shape [2, num_edges] where edge_index[0] 
              contains source node indices and edge_index[1] contains target node indices.
    """
    # 1. Calculate Nuclei Features
    nuclei_records = []
    nuclei_centroids = []

    for idx, nuc in enumerate(valid_nuclei_data):
        area = nuc['area']
        perimeter = nuc['perimeter']
        
        # Calculate average intensity
        coords = nuc['coords']
        avg_intensity = int_image[coords[:, 0], coords[:, 1]].mean()
        
        # Circularity formula: 4 * pi * (Area / Perimeter^2)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        
        nuclei_records.append({
            'node_id': idx,
            'circularity': circularity,
            'eccentricity': nuc['eccentricity'],
            'area': area,
            'average_intensity': avg_intensity,
            'major_axis_length': nuc['major_axis_length'],
            'minor_axis_length': nuc['minor_axis_length']
        })
        nuclei_centroids.append(nuc['centroid'])
        
    nuclei_df = pd.DataFrame(nuclei_records, columns=[
        'node_id', 'circularity', 'eccentricity', 'area', 'average_intensity',
        'major_axis_length', 'minor_axis_length'
    ])
    
    avg_nucleus_length = np.mean([n['major_axis_length'] for n in valid_nuclei_data]) if valid_nuclei_data else 1.0

    # 2. Calculate Paths Features
    path_records = []
    source_nodes = []
    target_nodes = []
    num_nuclei = len(valid_nuclei_data)
    
    if num_nuclei >= 2:
        masked_int_image = int_image.copy()
        masked_int_image[binary_mask_filled] = 0
        
        for i in range(num_nuclei):
            for j in range(i + 1, num_nuclei):
                c1 = valid_nuclei_data[i]['centroid']
                c2 = valid_nuclei_data[j]['centroid']
                
                dy = c2[0] - c1[0]
                dx = c2[1] - c1[1]
                dist = np.sqrt(dy**2 + dx**2)
                
                if dist == 0: continue
                
                normalized_length = dist / avg_nucleus_length
                
                # Calculate relative angle between each nucleus and the path (edge)
                path_angle = np.arctan2(dx, dy)
                angle1 = valid_nuclei_data[i]['orientation']
                angle2 = valid_nuclei_data[j]['orientation']
                angle_diff = abs(angle1 - angle2) % np.pi
                relative_angle = np.pi - angle_diff if angle_diff > np.pi / 2 else angle_diff
                
                diff1 = abs(path_angle - angle1) % np.pi
                node1_angle_diff = np.pi - diff1 if diff1 > np.pi / 2 else diff1
                
                diff2 = abs(path_angle - angle2) % np.pi
                node2_angle_diff = np.pi - diff2 if diff2 > np.pi / 2 else diff2

                profile = profile_line(masked_int_image, c1, c2, linewidth=path_width, mode='constant', cval=0)
                valid_profile = profile[profile > 0]
                mean_intensity = np.mean(valid_profile) if len(valid_profile) > 0 else 0.0
                
                path_records.append({
                    'source_node': i,
                    'target_node': j,
                    'average_intensity': mean_intensity,
                    'length': normalized_length,
                    'node1_angle_diff': node1_angle_diff / (np.pi / 2),
                    'node2_angle_diff': node2_angle_diff / (np.pi / 2),
                    'min_diff_angle': min(node1_angle_diff, node2_angle_diff) / (np.pi / 2),
                    'relative_angle': relative_angle / (np.pi / 2)
                })
                source_nodes.append(i)
                target_nodes.append(j)

    paths_df = pd.DataFrame(path_records, columns=[
        'source_node', 'target_node', 'average_intensity', 'length', 'node1_angle_diff', 'node2_angle_diff', 'min_diff_angle', 'relative_angle'
    ])
    
    edge_index = [source_nodes, target_nodes]
    
    if show_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(int_image, cmap='gray')
        
        # Draw edges
        if show_edge:
            for u, v in zip(source_nodes, target_nodes):
                y1, x1 = valid_nuclei_data[u]['centroid']
                y2, x2 = valid_nuclei_data[v]['centroid']
                plt.plot([x1, x2], [y1, y2], color='cyan', alpha=0.4, linewidth=1, zorder=1)
            
        # Draw colored nodes with IDs
        cmap = plt.cm.tab20
        for idx, nuc in enumerate(valid_nuclei_data):
            y, x = nuc['centroid']
            color = cmap(idx % 20)
            plt.scatter(x, y, color=color, s=150, edgecolors='white', zorder=2)
            plt.text(x, y, str(idx), color='black', fontsize=9, ha='center', va='center', 
                     fontweight='bold', zorder=3)
            
        plt.title('Extracted Graph: Nodes and IDs')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return nuclei_df, nuclei_centroids, paths_df, edge_index