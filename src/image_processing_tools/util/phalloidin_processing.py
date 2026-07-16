import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from image_processing_tools.util.visualize import invert_image_colors
from scipy.ndimage import gaussian_filter
    
def get_otsu_edge_map(image: np.ndarray) -> np.ndarray:
    """
    Applies Otsu's thresholding to an input array (e.g., an edge magnitude map)
    to return a binary edge map.

    Args:
        image (np.ndarray): Input array. Can be float or integer type.

    Returns:
        np.ndarray: Binary map (0s and 1s) where 1 represents the edges/foreground.
    """
    # OpenCV's Otsu implementation requires an 8-bit single-channel input image.
    if image.dtype != np.uint8:
        # Normalize the image to the 0-255 range for 8-bit conversion
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            # Scale to 0-255
            img_8u = (255 * (image - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            # Handle constant image case
            img_8u = np.zeros_like(image, dtype=np.uint8)
    else:
        img_8u = image

    # Apply Otsu's thresholding
    # cv2.THRESH_OTSU tells OpenCV to calculate the optimal threshold.
    # The explicit threshold value (0) is ignored.
    # maxval is set to 1 so the resulting binary map contains 0s and 1s.
    otsu_thresh_val, binary_map = cv2.threshold(
        img_8u, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary_map

def add_gradient_arrows_to_ax(
    ax: plt.Axes,
    gx: np.ndarray,
    gy: np.ndarray,
    step: int = 20,
    arrow_scale: float = 1.0,
    arrow_width: float = 0.005,
    alpha: float = 0.8,
    rotate_90: bool = False,
    show_arrowheads: bool = False,
    uniform_length: bool = True,
    color_by_magnitude: bool = True,
    flip_opposing_vectors: bool = True,
    use_log_magnitude: bool = False,
    colormap: str = 'viridis',
    clip_magnitude: bool = False
):
    """
    Overlays arrows representing gradient direction and magnitude on a matplotlib axis.

    Args:
        ax (plt.Axes): The matplotlib axis to draw on.
        gx (np.ndarray): Gradient in x direction.
        gy (np.ndarray): Gradient in y direction.
        step (int): Spacing between arrows (in pixels).
        arrow_scale (float): Scaling factor for arrow length.
        arrow_width (float): Width of the arrow shaft.
        alpha (float): Transparency of the arrows.
        rotate_90 (bool): If True, rotates the arrows by 90 degrees.
        show_arrowheads (bool): If True, draws arrow heads. Defaults to False (lines only).
        uniform_length (bool): If True, all arrows have the same length.
        color_by_magnitude (bool): If True, colors arrows by gradient magnitude.
        flip_opposing_vectors (bool): If True, flips vectors in the lower half-plane to point to the upper half-plane, treating them as orientations.
        use_log_magnitude (bool): If True, uses log of magnitude for coloring.
        colormap (str): The name of the colormap to use if color_by_magnitude is True.
        clip_magnitude (bool): If True, clips the magnitude to the 0.5-99.5 percentile range before coloring.
    """
    h, w = gx.shape
    # Create grid coordinates
    y, x = np.mgrid[step//2:h:step, step//2:w:step]
    
    # Sample gradients
    u = gx[step//2:h:step, step//2:w:step].copy()
    v = gy[step//2:h:step, step//2:w:step].copy()
    
    # Calculate magnitude
    magnitude = np.sqrt(u**2 + v**2)
    
    # Add a small amount to magnitude to avoid division by zero and log(0)
    magnitude = magnitude + 1e-9
    
    if rotate_90:
        # Rotate vectors (u, v) by 90 degrees: (x, y) -> (-y, x)
        u_rot = -v
        v_rot = u
        u, v = u_rot, v_rot
        
    if flip_opposing_vectors:
        # Flip opposing vectors so they all point into the upper half-plane (v>=0)
        mask = v < 0 # Identifies all Q3 and Q4 vectors
        u[mask] = -u[mask] # Flip their x
        v[mask] = -v[mask] # Flip their y

    mag_for_display = magnitude.copy()
    if clip_magnitude:
        vmin = np.percentile(mag_for_display, 0.5)
        vmax = np.percentile(mag_for_display, 99.5)
        mag_for_display = np.clip(mag_for_display, vmin, vmax)

    if color_by_magnitude:
        if use_log_magnitude:
            mag_for_color = np.log(mag_for_display)
        else:
            mag_for_color = mag_for_display
        norm = colors.Normalize(vmin=mag_for_color.min(), vmax=mag_for_color.max())
        cmap = plt.get_cmap(colormap)
        arrow_colors = cmap(norm(mag_for_color))
    else:
        # Calculate angle for coloring (in degrees)
        angle = np.arctan2(v, u) * (180 / np.pi)

        if flip_opposing_vectors:
            # Normalize angle to 0-1 for Hue. Since we flipped vectors to the
            # upper half-plane, the angle is in [0, 180].
            hue = angle / 180.0
        else:
            # Normalize full 360-degree angle to 0-1 for Hue
            hue = (angle + 180) / 360.0

        saturation = np.ones_like(hue)

        # Map Value (brightness) to magnitude, respecting the log scale option
        if use_log_magnitude:
            mag_for_value = np.log(mag_for_display)
        else:
            mag_for_value = mag_for_display
        
        # Normalize magnitude to 0-1 range for Value
        min_mag, max_mag = mag_for_value.min(), mag_for_value.max()
        value = (mag_for_value - min_mag) / (max_mag - min_mag) if max_mag > min_mag else np.ones_like(hue)
        
        hsv = np.stack((hue, saturation, value), axis=-1)
        arrow_colors = colors.hsv_to_rgb(hsv)
    
    if uniform_length:
        # Normalize vectors to unit length and scale uniformly
        u = u / magnitude
        v = v / magnitude
        fixed_length = step * arrow_scale * 0.8
        u *= fixed_length
        v *= fixed_length
    else:
        # Scale vector length by magnitude. The longest arrow will have a length
        # proportional to `step * arrow_scale`.
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            scaling_factor = (step * arrow_scale * 0.8) / max_magnitude
            u *= scaling_factor
            v *= scaling_factor

    headwidth = 3 if show_arrowheads else 0
    headlength = 5 if show_arrowheads else 0
    headaxislength = 4.5 if show_arrowheads else 0

    # Flatten arrow_colors to match the flattened u and v arrays that quiver expects
    arrow_colors = arrow_colors.reshape(-1, arrow_colors.shape[-1])

    ax.quiver(x, y, u, v, color=arrow_colors, angles='xy', scale_units='xy', scale=1, 
              width=arrow_width, pivot='mid', alpha=alpha, 
              headwidth=headwidth, headlength=headlength, headaxislength=headaxislength)

    if color_by_magnitude:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Log Magnitude' if use_log_magnitude else 'Magnitude')
    else:
        # Add a colorbar for the angle when coloring by direction
        cmap_angle = plt.get_cmap('hsv')
        if flip_opposing_vectors:
            norm_angle = colors.Normalize(vmin=0, vmax=180)
            cbar_label = 'Orientation (degrees)'
        else:
            norm_angle = colors.Normalize(vmin=-180, vmax=180)
            cbar_label = 'Direction (degrees)'
        sm_angle = plt.cm.ScalarMappable(cmap=cmap_angle, norm=norm_angle)
        cbar_angle = plt.colorbar(sm_angle, ax=ax, fraction=0.046, pad=0.04)
        cbar_angle.set_label(cbar_label)

def calculate_gradient_direction_color(gx: np.ndarray, gy: np.ndarray, flip_opposing_vectors: bool = True) -> np.ndarray:
    """
    Calculates gradient directions and represents them as a color image (HSV -> RGB).
    Low magnitude areas appear white (low saturation), high magnitude areas appear colorful.
    The direction is rotated by 90 degrees (orthogonal to original).
    
    Args:
        gx (np.ndarray): Gradient in x direction.
        gy (np.ndarray): Gradient in y direction.
        flip_opposing_vectors (bool): If True, treats opposing vectors as the same orientation (180-degree symmetry).
        
    Returns:
        np.ndarray: RGB image representing gradient directions.
    """
    # Calculate magnitude and angle
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) # Degrees -180 to 180
    
    # Rotate angle by 90 degrees
    angle += 90
    
    # Normalize angle to [0, 360) range
    angle = angle % 360
    
    if flip_opposing_vectors:
        # Fold the 360-degree range into a 180-degree range for orientation
        angle = angle % 180
        # Map 0-180 degrees to 0-180 values for Hue (uses full color cycle for orientation)
        hue = angle.astype(np.uint8)
    else:
        # Map 0-360 degrees to 0-180 values for Hue (OpenCV standard)
        hue = (angle / 2).astype(np.uint8)
    
    # Value is set to maximum (bright) so the background is white
    value = np.ones_like(hue) * 255
    # value = (magnitude / magnitude.max() * 255).astype(np.uint8) # alternatively map value (brightness) to magnitude
    
    # Saturation is set based on normalized magnitude
    if magnitude.max() > 0:
        saturation = (magnitude / magnitude.max() * 255).astype(np.uint8)
    else:
        saturation = np.zeros_like(hue)
        
    # Create HSV image
    hsv = cv2.merge([hue, saturation, value])
    
    # Convert to RGB for display
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def plot_gradient_magnitude_histogram(gx: np.ndarray, gy: np.ndarray, bins: int = 100, use_log_magnitude: bool = False, log_frequency: bool = True, clip_magnitude: bool = False):
    """
    Displays a histogram of gradient magnitudes.
    
    Args:
        gx (np.ndarray): Gradient in x direction.
        gy (np.ndarray): Gradient in y direction.
        bins (int): Number of bins for the histogram.
        use_log_magnitude (bool): If True, plots the histogram of log(magnitude).
        log_frequency (bool): If True, uses a log scale for the frequency (y-axis).
        clip_magnitude (bool): If True, clips the magnitude to the 0.5-99.5 percentile range.
    """
    magnitude = np.sqrt(gx**2 + gy**2)
    
    if clip_magnitude:
        vmin = np.percentile(magnitude, 0.5)
        vmax = np.percentile(magnitude, 99.5)
        magnitude = np.clip(magnitude, vmin, vmax)
    
    if use_log_magnitude:
        # Add small epsilon to avoid log(0)
        magnitude = np.log(magnitude + 1e-9)
        xlabel = "Log(Magnitude)"
        title_suffix = "(Log Transformed)"
    else:
        xlabel = "Magnitude"
        title_suffix = ""

    plt.figure(figsize=(10, 4))
    plt.hist(magnitude.ravel(), bins=bins, log=log_frequency, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of Gradient Magnitudes {title_suffix}")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency" + (" (Log Scale)" if log_frequency else ""))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_gradient_field_on_empty_axis(ax, gx, gy, **kwargs):
    """
    Plots the gradient field on an empty axis with inverted Y to match image coordinates.
    
    Args:
        ax (plt.Axes): The matplotlib axis to draw on.
        gx (np.ndarray): Gradient in x direction.
        gy (np.ndarray): Gradient in y direction.
        kwargs: Additional arguments for add_gradient_arrows_to_ax.
    """
    h, w = gx.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0) # Invert Y to match image coordinates
    ax.set_aspect('equal')
    add_gradient_arrows_to_ax(ax, gx, gy, **kwargs)
    ax.set_title('Gradient Field Only')

def plot_3d_gradient_magnitude_plotly(gx: np.ndarray, gy: np.ndarray, downsample: int = 2, flip_opposing_vectors: bool = True, smooth_sigma: float = 2.0, save_path: str = None, use_log_magnitude: bool = False, show_plot: bool = True, smooth_angle_sigma: float = 0.0, clip_magnitude: bool = False):
    """
    Creates an interactive 3D surface plot using Plotly.
    This is often more robust in Jupyter notebooks than matplotlib's 3D backend.
    
    Args:
        gx (np.ndarray): Gradient in x direction.
        gy (np.ndarray): Gradient in y direction.
        downsample (int): Factor to downsample the image for faster plotting.
        flip_opposing_vectors (bool): If True, treats opposing vectors as the same orientation.
        smooth_sigma (float): Sigma for Gaussian smoothing of the magnitude surface.
        save_path (str): If provided, saves the interactive plot as an HTML file.
        use_log_magnitude (bool): If True, uses log of magnitude for height.
        show_plot (bool): If True, displays the plot in the notebook. Defaults to True.
        smooth_angle_sigma (float): Sigma for Gaussian smoothing of the orientation field.
        clip_magnitude (bool): If True, clips the magnitude to the 0.5-99.5 percentile range.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Error: Plotly is not installed. Please install it using: pip install plotly")
        return

    # Downsample
    gx_small = gx[::downsample, ::downsample]
    gy_small = gy[::downsample, ::downsample]
    
    # Magnitude (Z)
    Z = np.sqrt(gx_small**2 + gy_small**2)
    
    if clip_magnitude:
        vmin = np.percentile(Z, 0.5)
        vmax = np.percentile(Z, 99.5)
        Z = np.clip(Z, vmin, vmax)
    
    if use_log_magnitude:
        Z = np.log(Z + 1)
    
    if smooth_sigma > 0:
        Z = gaussian_filter(Z, sigma=smooth_sigma)
    
    # Orientation (Color)
    if smooth_angle_sigma > 0:
        if flip_opposing_vectors:
            # Use structure tensor approach to smooth orientation (0-180)
            # This prevents cancellation of opposing vectors
            jxx = gx_small**2
            jyy = gy_small**2
            jxy = gx_small * gy_small
            
            jxx = gaussian_filter(jxx, sigma=smooth_angle_sigma)
            jyy = gaussian_filter(jyy, sigma=smooth_angle_sigma)
            jxy = gaussian_filter(jxy, sigma=smooth_angle_sigma)
            
            angle = 0.5 * np.arctan2(2*jxy, jxx - jyy) * (180 / np.pi)
        else:
            gx_smooth = gaussian_filter(gx_small, sigma=smooth_angle_sigma)
            gy_smooth = gaussian_filter(gy_small, sigma=smooth_angle_sigma)
            angle = np.arctan2(gy_smooth, gx_smooth) * (180 / np.pi)
    else:
        angle = np.arctan2(gy_small, gx_small) * (180 / np.pi)
        
    angle += 90
    angle = angle % 360
    
    if flip_opposing_vectors:
        angle = angle % 180
        cmax = 180
    else:
        cmax = 360

    # Create figure
    fig = go.Figure(data=[go.Surface(z=Z, surfacecolor=angle, colorscale='HSV', cmin=0, cmax=cmax)])
    
    fig.update_layout(
        title=f"Gradient Magnitude (Height) vs Orientation (Color) - Downsample {downsample}x",
        scene=dict(
            xaxis_title='X',
            yaxis=dict(title='Y', autorange="reversed"), # Reverse Y to match image coordinates
            zaxis_title='Log Magnitude' if use_log_magnitude else 'Magnitude',
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive 3D plot to {save_path}")
    
    if show_plot:
        fig.show()

def process_actin_fibers(img, edge_method='sobel', show_plot=False, **kwargs):
    """
    Process actin fibers with denoising, edge detection, and multiscale product.
    
    Args:
        img (np.ndarray): Input 2D image.
        edge_method (str): 'mexican_hat' (2nd derivative), 'sobel' (1st derivative), or 'haar'.
        kwargs: Additional arguments for the add_gradient_arrows_to_ax function.
        
    Returns:
        gx, gy: 1st order derivatives in x and y directions.
    """

    # Default arguments for arrow visualization (can be overridden by kwargs)
    arrow_kwargs = {
        'step': 50,
        'rotate_90': True,
        'arrow_scale': 1.0,
        'alpha': 0.8,
        'colormap': 'cividis',
        'use_log_magnitude': True,
        'flip_opposing_vectors': True,
        'clip_magnitude': False
    }
    arrow_kwargs.update(kwargs)

    # --- PART 1: DENOISING (Sym8) ---
    wavelet_denoise = 'sym8'
    level = 3
    coeffs = pywt.wavedec2(img, wavelet_denoise, level=level)
    
    sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(img.size))
    
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in coeffs[i]))
    
    img_denoised = pywt.waverec2(new_coeffs, wavelet_denoise)

    # --- PART 2: EDGE DETECTION ---
    
    gx, gy, edges = None, None, None
    
    if edge_method == 'mexican_hat':
        edges = cv2.Laplacian(img_denoised, cv2.CV_64F)
        edges = np.abs(edges)
        gx = cv2.Sobel(img_denoised, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_denoised, cv2.CV_64F, 0, 1, ksize=3)
        
    elif edge_method == 'sobel':
        gx = cv2.Sobel(img_denoised, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_denoised, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(gx**2 + gy**2)

    elif edge_method == 'haar':
        coeffs = pywt.dwt2(img_denoised, 'haar')
        cA, (cH, cV, cD) = coeffs
        
        h, w = img.shape
        cH = cv2.resize(cH, (w, h), interpolation=cv2.INTER_LINEAR)
        cV = cv2.resize(cV, (w, h), interpolation=cv2.INTER_LINEAR)
        cD = cv2.resize(cD, (w, h), interpolation=cv2.INTER_LINEAR)
        
        edges = np.sqrt(cH**2 + cV**2 + cD**2)
        gx = cH
        gy = cV

    else:
        raise ValueError("edge_method must be 'mexican_hat', 'sobel', or 'haar'")

    # Calculate gradient directions for visualization
    grad_colors = calculate_gradient_direction_color(gx, gy, flip_opposing_vectors=arrow_kwargs.get('flip_opposing_vectors', True))

    # Get binary edges
    edges_binary = get_otsu_edge_map(edges)

    # # --- PART 3: MULTISCALE PRODUCT (The Correlation Step) ---
    # # Goal: enhances edges and suppresses noise because edges would have strong coefficients in multiple scales. Their multiplication will then enhance the edge. If one of them is noise and have low wavelet coefficients, the multiplication will shrink it.
    # # Doesn't work well. Not used for now.
    
    # coeffs_mp = pywt.wavedec2(img_denoised, 'bior3.5', level=level)
    
    # def get_mag(level_coeffs):
    #     LH, HL, HH = level_coeffs
    #     return np.sqrt(LH**2 + HL**2 + HH**2)

    # mag_l1 = get_mag(coeffs_mp[-1])
    # mag_l2 = get_mag(coeffs_mp[-2])
    
    # mag_l2_resized = cv2.resize(mag_l2, (mag_l1.shape[1], mag_l1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # multiscale_prod = mag_l1 * mag_l2_resized
    
    # if np.issubdtype(img.dtype, np.integer):
    #     info = np.iinfo(img.dtype)
    #     min_val = multiscale_prod.min()
    #     max_val = multiscale_prod.max()
        
    #     if max_val > min_val:
    #         multiscale_prod = (multiscale_prod - min_val) / (max_val - min_val) * info.max
    #     else:
    #         multiscale_prod = np.zeros_like(multiscale_prod)
            
    #     multiscale_prod = multiscale_prod.astype(img.dtype)
    
    # multiscale_prod_binary = get_otsu_edge_map(multiscale_prod)    

    # --- VISUALIZATION ---
    if show_plot:
        def safe_invert(im):
            try: return invert_image_colors(im)
            except NameError: return 255 - im if im.dtype==np.uint8 else im
            
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        axes[0].imshow(img, cmap='gray'); axes[0].set_title('Original')
        axes[1].imshow(img_denoised, cmap='gray'); axes[1].set_title('Denoised (Sym8)')
        add_gradient_arrows_to_ax(axes[1], gx, gy, **arrow_kwargs)
        axes[2].imshow(grad_colors); axes[2].set_title('Gradient Directions')
        
        axes[3].imshow(safe_invert(edges_binary.astype(bool)), cmap='gray', interpolation = 'nearest')
        # add_gradient_arrows_to_ax(axes[3], gx, gy, **arrow_kwargs)
        axes[3].set_title(f'Edge Detection ({edge_method})')
        
        # Multiscaleproduct plot not used for now
        # axes[4].imshow(safe_invert(multiscale_prod_binary.astype(bool)), cmap='gray', interpolation = 'nearest')
        # axes[4].set_title('Multiscale Product')
        
        # Plot 5: Gradient Field on Empty Axes
        plot_gradient_field_on_empty_axis(axes[4], gx, gy, **arrow_kwargs)
        
        # Turn off axis lines/ticks for all plots
        for ax in axes[:4]: # Keep frame for the empty plot if desired, or turn off for all
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    return gx,gy

def fit_gradient_magnitude_gmm(gx: np.ndarray, gy: np.ndarray, n_components: int = 2, use_log_magnitude: bool = True, clip_magnitude: bool = False, show_plot: bool = True, prob_colormap: str = 'magma'):
    """
    Fits a Gaussian Mixture Model to the gradient magnitudes to separate signal from background.
    
    Args:
        gx (np.ndarray): Gradient in x direction.
        gy (np.ndarray): Gradient in y direction.
        n_components (int): Number of Gaussian components to fit.
        use_log_magnitude (bool): If True, uses log of magnitude for fitting.
        clip_magnitude (bool): If True, clips the magnitude to the 0.5-99.5 percentile range.
        show_plot (bool): If True, displays the histogram fit and probability maps.
        prob_colormap (str): Colormap for the probability map. Use 'uncertainty' for a custom map highlighting 0.5 prob.
        
    Returns:
        np.ndarray: Probability maps of shape (H, W, n_components), sorted by component mean (low to high).
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("Error: scikit-learn is not installed. Please install it using: pip install scikit-learn")
        return None

    magnitude = np.sqrt(gx**2 + gy**2)
    
    if clip_magnitude:
        vmin = np.percentile(magnitude, 0.5)
        vmax = np.percentile(magnitude, 99.5)
        magnitude = np.clip(magnitude, vmin, vmax)
    
    if use_log_magnitude:
        # Add small epsilon to avoid log(0)
        data = np.log(magnitude + 1e-9)
        xlabel = "Log(Magnitude)"
    else:
        data = magnitude
        xlabel = "Magnitude"
        
    # Flatten for GMM
    X = data.reshape(-1, 1)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Get probabilities
    probs = gmm.predict_proba(X)
    
    # Sort components by mean (so index 0 is background/low, index -1 is signal/high)
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    
    probs_sorted = probs[:, sorted_indices]
    means_sorted = means[sorted_indices]
    weights_sorted = gmm.weights_[sorted_indices]
    covariances_sorted = gmm.covariances_.flatten()[sorted_indices]
    
    # Reshape back to image
    h, w = gx.shape
    prob_maps = probs_sorted.reshape(h, w, n_components)
    
    if show_plot:
        from scipy.stats import norm
        
        # Plot 1: Histogram and GMM Fit
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        x_grid = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x_grid)
        pdf = np.exp(logprob)
        
        plt.hist(X.flatten(), bins=100, density=True, alpha=0.5, color='gray', label='Data Histogram')
        plt.plot(x_grid, pdf, '-k', label='GMM Sum')
        
        # Plot individual components
        comp_colors = plt.cm.viridis(np.linspace(0, 1, n_components))
        
        for i in range(n_components):
            mean = means_sorted[i]
            var = covariances_sorted[i]
            weight = weights_sorted[i]
            pdf_comp = weight * norm.pdf(x_grid, mean, np.sqrt(var))
            plt.plot(x_grid, pdf_comp, '--', color=comp_colors[i], label=f'Comp {i+1} (Mean={mean:.2f})')
            
        plt.title(f'GMM Fit (n={n_components}) on {xlabel}')
        plt.xlabel(xlabel)
        plt.ylabel('Density')
        plt.legend()
        
        # Plot 2: Probability Maps
        plt.subplot(1, 2, 2)
        # Show the highest magnitude component as "Signal Probability"
        
        cmap_to_use = prob_colormap
        if prob_colormap == 'uncertainty':
            # Custom colormap: Black (0) -> Yellow (0.5) -> Black (1)
            # Highlights uncertain regions where probability is around 0.5
            cmap_to_use = colors.LinearSegmentedColormap.from_list('uncertainty', [(0.0, 'black'), (0.5, 'yellow'), (1.0, 'black')])
            
        plt.imshow(prob_maps[..., -1], cmap=cmap_to_use, vmin=0, vmax=1)
        plt.title(f'Probability of Highest Magnitude Component\n(Signal Probability)')
        plt.colorbar(label='Probability')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Optional: If n > 2, show all maps in a separate figure
        if n_components > 2:
            fig, axes = plt.subplots(1, n_components, figsize=(4*n_components, 4))
            for i in range(n_components):
                axes[i].imshow(prob_maps[..., i], cmap='gray', vmin=0, vmax=1)
                axes[i].set_title(f'Component {i+1} Prob\n(Mean={means_sorted[i]:.2f})')
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()

    return prob_maps