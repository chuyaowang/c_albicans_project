import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter, grey_erosion, grey_dilation
from skimage.transform import resize
import skimage.data

class BackgroundSubtracter:
    def __init__(self):
        pass

    def rolling_ball_background(self, image, radius, light_background=False,
                                separate_colors=False, use_paraboloid=True, debug=False):
        """
        Replicates modern ImageJ's BackgroundSubtracter for Float images.
        Uses the 'Sliding Paraboloid' approach which is robust against halos.
        
        Parameters
        ----------
        image : ndarray
            Input image (float64, 0-1 range).
        radius : float
            Rolling ball radius in pixels.
        light_background : bool
            If True, treats image as light background with dark objects.
        use_paraboloid : bool
            If True, uses the sliding paraboloid (faster, robust). If False, uses the classic rolling ball.
        debug : bool
            If True, plots debug views.
        """
        img = image.astype(np.float64)
        
        # 1. Invert if Light Background
        # The algorithm works on "bright objects on dark background"
        if light_background:
            img = 1.0 - img
            
        # 2. Shrink Image (Speed optimization)
        # ImageJ logic: shrink if radius is large to save time
        shrink_factor = self._get_shrink_factor(radius)
        
        if shrink_factor > 1:
            # Shrink by downsampling (using mean/box filter approach or simple slicing)
            # ImageJ uses a specific downsampling that we approximate here
            # We use a simple decimation after smoothing to avoid aliasing
            img_smooth = uniform_filter(img, size=shrink_factor)
            small_img = img_smooth[::shrink_factor, ::shrink_factor]
            scale_radius = radius / shrink_factor
        else:
            small_img = img
            scale_radius = radius

        # 3. Apply Sliding Paraboloid Algorithm
        # This is the core logic replacement
        if use_paraboloid:
            background_small = self._sliding_paraboloid(small_img, scale_radius)
        else:
            background_small = self._rolling_ball_2d(small_img, scale_radius)
        
        # 4. Enlarge Background to Original Size
        if shrink_factor > 1:
            # Bilinear upscaling
            background = resize(background_small, img.shape, order=1, 
                                mode='edge', preserve_range=True)
        else:
            background = background_small

        # 5. Subtract Background
        result = img - background
        
        # Clip negative values (essential for float images)
        result = np.clip(result, 0, 1)

        # 6. Invert Output if needed
        if light_background:
            final_output = 1.0 - result
            final_bg = 1.0 - background
            debug_img = img # Show inverted 'working' image for consistency
        else:
            final_output = result
            final_bg = background
            debug_img = img

        if debug:
            self._plot_debug(debug_img, background, result)

        return final_output

    def _get_shrink_factor(self, radius):
        """Matches ImageJ's shrinkage logic."""
        # Simple approximation of ImageJ logic
        if radius <= 10: return 1
        if radius <= 30: return 2
        if radius <= 100: return 4
        return 8

    def _sliding_paraboloid(self, img, radius):
        """
        Implements the 'Sliding Paraboloid' morphological opening.
        Instead of a ball, we slide a parabola p(x) = curvature * x^2.
        
        The 'curvature' is derived such that the parabola touches the 
        background (0) at distance 'radius' from the apex (height).
        However, for float images, ImageJ often treats the 'height' 
        of the ball as related to the pixel intensity range.
        
        Modern ImageJ simplified: For each pixel, find the minimum 
        value of (pixel - parabola) in the neighborhood. Then find 
        the maximum of those minima.
        """
        rows, cols = img.shape
        background = np.zeros_like(img)
        
        # Define the Parabola
        # The parabola function is: z = a * r^2
        # We need to determine 'a' (curvature).
        # In ImageJ's FloatProcessor logic, the ball acts "as if" the 
        # intensity range is mapped to the spatial radius. 
        # A common heuristic for float images [0,1] is that the "ball height" 
        # is relative to the max intensity (1.0).
        
        # Correct curvature 'a' calculation:
        # If we want the parabola to drop by '1.0' (full intensity range) 
        # over the distance 'radius':
        # 1.0 = a * radius^2  =>  a = 1.0 / (radius^2)
        curvature = 1.0 / (radius * radius)

        # We can separate the 2D paraboloid into two 1D passes (X and Y).
        # This is a property of the paraboloid (separable kernel) and 
        # makes it O(N) instead of O(N^2).
        
        # Pass 1: X-direction
        # For each row, calculate the morphological opening with 1D parabola
        temp_img = np.zeros_like(img)
        for r in range(rows):
            temp_img[r, :] = self._rolling_parabola_1d(img[r, :], radius, curvature)
            
        # Pass 2: Y-direction (on the result of Pass 1)
        for c in range(cols):
            background[:, c] = self._rolling_parabola_1d(temp_img[:, c], radius, curvature)
            
        return background

    def _rolling_ball_2d(self, img, radius):
        """
        Implements the classic 'Rolling Ball' algorithm using 2D morphological opening.
        Uses a spherical structuring element.
        """
        # Radius calculation
        r_int = int(np.ceil(radius))
        
        # Create grid for the ball structure
        x, y = np.mgrid[-r_int:r_int+1, -r_int:r_int+1]
        dist_sq = x**2 + y**2
        mask = dist_sq <= radius**2
        
        # Define Ball Structure (Hemisphere)
        # z = sqrt(R^2 - r^2)
        # Scale z by 1/255 to match ImageJ 8-bit behavior on float [0-1]
        scale = 1.0 / 255.0
        structure = np.zeros_like(dist_sq, dtype=np.float64)
        structure[mask] = np.sqrt(np.maximum(0, radius**2 - dist_sq[mask])) * scale
        
        # Apply Morphological Opening (Erosion + Dilation)
        # Erosion: min(img - structure)
        eroded = grey_erosion(img, structure=structure, footprint=mask)
        
        # Dilation: max(eroded + structure)
        background = grey_dilation(eroded, structure=structure, footprint=mask)
        
        return background

    def _rolling_parabola_1d(self, vector, radius, curvature):
        """
        1D Grayscale Opening with a Parabola.
        Operation: (f - b) + b  where b is the parabola structuring element.
        This is Erosion followed by Dilation.
        
        Erosion: g(x) = min_{u} ( f(x+u) - b(u) )
        Dilation: h(x) = max_{u} ( g(x+u) + b(u) )
        """
        n = len(vector)
        erosion = np.zeros(n)
        dilation = np.zeros(n)
        
        # The window range 'u' goes from -radius to +radius
        r_int = int(np.ceil(radius))
        
        # Precompute parabola kernel for the window
        # b(u) = curvature * u^2
        u_vals = np.arange(-r_int, r_int + 1)
        b_vals = curvature * (u_vals ** 2)
        
        # 1. EROSION (Min filter with subtraction)
        # Naive implementation O(N*R). Can be optimized, but N*R is fine for 1D.
        for i in range(n):
            # Extract local window (handling boundaries)
            start = max(0, i - r_int)
            end = min(n, i + r_int + 1)
            
            # Map indices to kernel coordinates
            # if i=10, start=5, then kernel index corresponds to u = 5-10 = -5
            k_start = r_int + (start - i)
            k_end = r_int + (end - i)
            
            window_slice = vector[start:end]
            kernel_slice = b_vals[k_start:k_end]
            
            # g(i) = min( f(x+u) - b(u) )
            # Note: b(u) is "depth" of parabola. 
            # We want the parabola *under* the curve, so we *add* depth to find 
            # the apex? No, standard definition:
            # Structure element is 'b'. Erosion is min(f - b).
            # Parabola opens upwards B(u) = ku^2.
            # We are fitting it *underneath*, so we look for the highest apex.
            # Actually, standard gray opening is: 
            #   Erosion E[x] = min_u ( I[x+u] + P[u] ) where P is parabola opening UP.
            #   Dilation D[x] = max_u ( E[x+u] - P[u] )
            # Let's stick to the geometric intuition:
            # To fit a parabola P(u) = k*u^2 *below* the signal:
            # The apex at 'x' cannot be higher than (Signal[x+u] - k*u^2).
            # So Apex[x] = min_u (Signal[x+u] - k*u^2).
            
            val = window_slice + kernel_slice # Wait, kernel slice is positive k*u^2
            # We want min(Signal - Depth).
            # Depth at u is k*u^2. 
            # So val = window_slice + kernel_slice (if we treat kernel as negative shape?)
            # Let's use the formula: Apex = min(Signal[x+u] + Curvature*u^2) 
            # Wait, if Signal is 0 and u is nonzero, Apex should be negative? 
            # Yes. So we subtract curvature*u^2?
            # Correct logic: The parabola y = -k*u^2 (opening down) is pushed up.
            # We want max(Apex) such that Apex - k*u^2 <= Signal[x+u].
            # Apex <= Signal[x+u] + k*u^2.
            # So Apex = min(Signal[x+u] + k*u^2).
            
            erosion[i] = np.min(window_slice + kernel_slice)

        # 2. DILATION (Max filter with addition)
        # Reconstruct the surface from the apexes.
        # Surface[x] = max_u ( Apex[x+u] - k*u^2 )
        for i in range(n):
            start = max(0, i - r_int)
            end = min(n, i + r_int + 1)
            
            k_start = r_int + (start - i)
            k_end = r_int + (end - i)
            
            window_slice = erosion[start:end]
            kernel_slice = b_vals[k_start:k_end]
            
            # Surface = max(Apex - Depth)
            dilation[i] = np.max(window_slice - kernel_slice)
            
        return dilation

    def _plot_debug(self, img, bg, res):
        """Plotting helper."""
        mid_row = img.shape[0] // 2
        
        plt.figure(figsize=(14, 8))
        
        # Images
        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("1. Input (Working)")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(bg, cmap='gray')
        plt.title("2. Background")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(res, cmap='gray')
        plt.title("3. Result")
        plt.axis('off')
        
        # Profiles
        plt.subplot(2, 1, 2)
        p_img = img[mid_row, :]
        p_bg = bg[mid_row, :]
        
        plt.plot(p_img, color='black', label='Input', linewidth=1, alpha=0.8)
        plt.plot(p_bg, color='red', label='Background', linewidth=2, linestyle='--')
        
        # Fill the "result" area
        plt.fill_between(range(len(p_img)), p_img, p_bg, where=(p_img>p_bg),
                         color='blue', alpha=0.1, label='Difference (Result)')
        
        plt.title(f"1D Profile at Row {mid_row}")
        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Create the processor
    bs = BackgroundSubtracter()
    
    # Load Image (Coins is standard)
    image = skimage.data.coins()
    image = image.astype(np.float64) / 255.0
    
    # Run the NEW Sliding Paraboloid implementation
    print("Running Modern ImageJ Paraboloid Logic...")
    result = bs.rolling_ball_background(
        image, 
        radius=50, 
        light_background=False, 
        debug=True  # <--- Shows the requested plots
    )