import numpy as np
import scipy.fft
from skimage import exposure, filters, img_as_float
from scipy.ndimage import uniform_filter, grey_erosion, grey_dilation
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.transform import resize

class ImageJProcessor:
    """
    A modular image processing pipeline replicating ImageJ functionalities.
    Supports: CLAHE, Gaussian Blur, FFT Bandpass, Rolling Ball Background Subtraction, and Frangi Filtering.
    """

    def __init__(self, image):
        """
        Initialize with a 2D image (8-bit or 16-bit).
        """
        # Convert to float (0.0 to 1.0) for processing to avoid overflow/underflow
        self.original_image = image
        self.image = img_as_float(image).astype(np.float64)

        # Normalize if float input is out of [0, 1] range (e.g. unscaled float64 from preprocessing)
        if self.image.max() > 1.0:
            v_min, v_max = self.image.min(), self.image.max()
            if v_max - v_min > 1e-10:
                self.image = (self.image - v_min) / (v_max - v_min)
            else:
                self.image = np.zeros_like(self.image)

        self.history = ["Loaded Image"]

    def reset(self):
        """Revert image to original state."""
        self.image = img_as_float(self.original_image).astype(np.float64)
        if self.image.max() > 1.0:
            v_min, v_max = self.image.min(), self.image.max()
            if v_max - v_min > 1e-10:
                self.image = (self.image - v_min) / (v_max - v_min)
            else:
                self.image = np.zeros_like(self.image)
        self.history = ["Reset to Original"]

    def enhance_contrast_clahe(self, block_size=127, slope=3.0, bins=256):
        """
        Replicates ImageJ's 'Enhance Local Contrast (CLAHE)'.
        
        Parameters:
        - block_size (int): Size of the local region (ImageJ calls this 'Blocksize').
        - slope (float): Maximum contrast limit (ImageJ calls this 'Maximum Slope').
        - bins (int): Number of histogram bins (ImageJ defaults to 256).
        """
        print(f"Applying CLAHE (Block: {block_size}, Slope: {slope}, Bins: {bins})...")
        
        # 1. Calculate Kernel Size
        kernel_size = (block_size, block_size)
        
        # 2. Convert ImageJ 'Slope' to skimage 'Clip Limit'
        # ImageJ Slope is relative to the flat histogram height (1/bins).
        # Skimage Clip Limit is a normalized probability (0 to 1).
        # Formula: clip_limit = slope / nbins
        v_clip_limit = slope / bins
        
        # 3. Apply CLAHE
        self.image = exposure.equalize_adapthist(
            self.image, 
            kernel_size=kernel_size, 
            clip_limit=v_clip_limit, 
            nbins=bins
        )
        self.history.append("CLAHE")
        return self.image

    def gaussian_blur(self, sigma=2.0):
        """
        Replicates ImageJ's 'Gaussian Blur'.
        
        Parameters:
        - sigma (float): Radius of decay to exp(-0.5).
        """
        print(f"Applying Gaussian Blur (Sigma: {sigma})...")
        self.image = filters.gaussian(self.image, sigma=sigma, mode='reflect')
        self.history.append("Gaussian Blur")
        return self.image

    def fft_bandpass_filter(self, large_structures_down_to=40, small_structures_up_to=3, suppress_stripes='None', tolerance=0.05, autoscale=True):
        """
        Replicates ImageJ's 'FFT Bandpass Filter' EXACTLY.
        
        Corrections applied:
        1. Cutoff Frequency: Corrected to 1.0 / (2.0 * Size) based on ImageJ source code.
        2. Padding: Implemented 'Reflect' padding to next power of 2 (matches ImageJ's tileMirror).
        """
        print(f"Applying ImageJ-Style FFT (Large: {large_structures_down_to}, Small: {small_structures_up_to})...")
        
        original_rows, original_cols = self.image.shape
        
        # --- 1. Pad Image (ImageJ Logic) ---
        # ImageJ pads to the next Power of 2 that is >= 1.5x the dimensions
        target_dim = max(original_rows, original_cols)
        pad_size = 2
        while pad_size < 1.5 * target_dim:
            pad_size *= 2
            
        pad_r = pad_size - original_rows
        pad_c = pad_size - original_cols
        
        # Pad with reflection (ImageJ uses mirroring) to avoid edge artifacts
        padded_image = np.pad(self.image, 
                            ((0, pad_r), (0, pad_c)), 
                            mode='reflect')
        
        rows, cols = padded_image.shape
        
        # --- 2. Create Frequency Grid ---
        r_idx, c_idx = np.mgrid[0:rows, 0:cols]
        c_idx = c_idx - cols / 2
        r_idx = r_idx - rows / 2
        
        norm_y = r_idx / rows
        norm_x = c_idx / cols
        frequency = np.sqrt(norm_y**2 + norm_x**2)
        
        # --- 3. Gaussian Bandpass (Corrected Formula) ---
        # Formula derived from ImageJ Source: Fc = 1 / (2 * Size)
        
        # High Pass
        if large_structures_down_to > 0:
            f_cutoff_large = 1.0 / (2.0 * large_structures_down_to)
            high_pass_mask = 1.0 - np.exp(-(frequency / f_cutoff_large)**2)
        else:
            high_pass_mask = 1.0

        # Low Pass
        if small_structures_up_to > 0:
            f_cutoff_small = 1.0 / (2.0 * small_structures_up_to)
            low_pass_mask = np.exp(-(frequency / f_cutoff_small)**2)
        else:
            low_pass_mask = 1.0
            
        mask = high_pass_mask * low_pass_mask

        # --- 4. Stripe Suppression ---
        if suppress_stripes in ['Horizontal', 'Vertical', 'Both']:
            safe_freq = frequency.copy()
            safe_freq[safe_freq == 0] = 1.0 
            sigma = tolerance
            stripe_mask = np.ones_like(mask)

            if suppress_stripes in ['Horizontal', 'Both']:
                sin_angle_v = np.abs(norm_x) / safe_freq
                mask_v = 1.0 - np.exp(-(sin_angle_v**2) / (2 * sigma**2))
                stripe_mask *= mask_v

            if suppress_stripes in ['Vertical', 'Both']:
                sin_angle_h = np.abs(norm_y) / safe_freq
                mask_h = 1.0 - np.exp(-(sin_angle_h**2) / (2 * sigma**2))
                stripe_mask *= mask_h

            center_r, center_c = rows // 2, cols // 2
            stripe_mask[center_r-1:center_r+2, center_c-1:center_c+2] = 1.0
            mask *= stripe_mask
        
        # --- 5. FFT & Filtering ---
        fft_image = scipy.fft.fft2(padded_image)
        fft_shifted = scipy.fft.fftshift(fft_image)
        
        filtered_fft = fft_shifted * mask
        
        ifft_shifted = scipy.fft.ifftshift(filtered_fft)
        restored_padded = scipy.fft.ifft2(ifft_shifted).real
        
        # --- 6. Crop back to original size ---
        restored_image = restored_padded[:original_rows, :original_cols]
        
        # --- 7. Autoscale ---
        if autoscale:
            v_min, v_max = restored_image.min(), restored_image.max()
            if v_max - v_min > 1e-10:
                self.image = (restored_image - v_min) / (v_max - v_min)
            else:
                self.image = np.zeros_like(restored_image)
        else:
            self.image = np.clip(restored_image, 0, 1)

        self.history.append(f"FFT Bandpass (Stripes: {suppress_stripes})")
        return self.image

    def imagej_rolling_ball(self, radius=50, light_background=False, use_paraboloid=False):
        """
        Replicates modern ImageJ's BackgroundSubtracter for Float images.
        
        Parameters
        ----------
        radius : float
            Rolling ball radius in pixels.
        light_background : bool
            If True, treats image as light background with dark objects.
        use_paraboloid : bool
            If True, uses the sliding paraboloid (faster, robust). If False, uses the classic rolling ball.
        """
        print(f"Applying ImageJ Rolling Ball (Radius: {radius}, Paraboloid: {use_paraboloid})...")
        
        img = self.image.astype(np.float64)
        
        # 1. Invert if Light Background
        if light_background:
            img = 1.0 - img
            
        # 2. Shrink Image (Speed optimization)
        shrink_factor = self._get_shrink_factor(radius)
        
        if shrink_factor > 1:
            img_smooth = uniform_filter(img, size=shrink_factor)
            small_img = img_smooth[::shrink_factor, ::shrink_factor]
            scale_radius = radius / shrink_factor
        else:
            small_img = img
            scale_radius = radius

        # 3. Apply Algorithm
        if use_paraboloid:
            background_small = self._sliding_paraboloid(small_img, scale_radius)
        else:
            background_small = self._rolling_ball_2d(small_img, scale_radius)
        
        # 4. Enlarge Background to Original Size
        if shrink_factor > 1:
            background = resize(background_small, img.shape, order=1, 
                                mode='edge', preserve_range=True)
        else:
            background = background_small

        # 5. Subtract Background
        result = img - background
        
        # Clip negative values
        result = np.clip(result, 0, 1)

        # 6. Invert Output if needed
        if light_background:
            self.image = 1.0 - result
        else:
            self.image = result

        self.history.append(f"Rolling Ball (R={radius}, Para={use_paraboloid})")
        return self.image

    def _get_shrink_factor(self, radius):
        if radius <= 10: return 1
        if radius <= 30: return 2
        if radius <= 100: return 4
        return 8

    def _sliding_paraboloid(self, img, radius):
        rows, cols = img.shape
        background = np.zeros_like(img)
        curvature = 1.0 / (radius * radius)

        # Pass 1: X-direction
        temp_img = np.zeros_like(img)
        for r in range(rows):
            temp_img[r, :] = self._rolling_parabola_1d(img[r, :], radius, curvature)
            
        # Pass 2: Y-direction
        for c in range(cols):
            background[:, c] = self._rolling_parabola_1d(temp_img[:, c], radius, curvature)
            
        return background

    def _rolling_ball_2d(self, img, radius):
        r_int = int(np.ceil(radius))
        x, y = np.mgrid[-r_int:r_int+1, -r_int:r_int+1]
        dist_sq = x**2 + y**2
        mask = dist_sq <= radius**2
        
        scale = 1.0 / 255.0
        structure = np.zeros_like(dist_sq, dtype=np.float64)
        structure[mask] = np.sqrt(np.maximum(0, radius**2 - dist_sq[mask])) * scale
        
        eroded = grey_erosion(img, structure=structure, footprint=mask)
        background = grey_dilation(eroded, structure=structure, footprint=mask)
        
        return background

    def _rolling_parabola_1d(self, vector, radius, curvature):
        n = len(vector)
        erosion = np.zeros(n)
        dilation = np.zeros(n)
        r_int = int(np.ceil(radius))
        
        u_vals = np.arange(-r_int, r_int + 1)
        b_vals = curvature * (u_vals ** 2)
        
        # Erosion
        for i in range(n):
            start = max(0, i - r_int)
            end = min(n, i + r_int + 1)
            k_start = r_int + (start - i)
            k_end = r_int + (end - i)
            
            window_slice = vector[start:end]
            kernel_slice = b_vals[k_start:k_end]
            erosion[i] = np.min(window_slice + kernel_slice)

        # Dilation
        for i in range(n):
            start = max(0, i - r_int)
            end = min(n, i + r_int + 1)
            k_start = r_int + (start - i)
            k_end = r_int + (end - i)
            
            window_slice = erosion[start:end]
            kernel_slice = b_vals[k_start:k_end]
            dilation[i] = np.max(window_slice - kernel_slice)
            
        return dilation
    
    def frangi_imagej_ops(self, scales=[8, 10], spacing=(1, 1), beta=0.5, c=15):
        """
        Replicates the 'Process > Filters > Frangi Vesselness' Op in ImageJ using Gaussian Derivatives.
        
        Parameters:
        - scales (list): List of scales to check (e.g., [8, 10]).
        - spacing (tuple): Voxel dimensions (y, x). Default (1, 1).
        - beta: Frangi's Beta (blob suppression).
        - c: Frangi's C (contrast sensitivity). Default 15 assumes 8-bit range; auto-scaled for float.
        """
        print(f"Applying ImageJ-Ops Frangi (Scales: {scales}, Spacing: {spacing})...")
        # Output vesselness map
        max_vesselness = np.zeros_like(self.image)
        # Output scale map (records which scale gave the max response)
        scale_map = np.zeros_like(self.image)

        # Adjust 'c' for float images if it looks like an 8-bit parameter
        # ImageJ 'c' is typically around 10-50 for 8-bit images.
        # For 0-1 float images, it should be scaled down to match the intensity range.
        c_scaled = c
        if self.image.max() <= 1.0 and c > 1.0:
            c_scaled = c / 255.0

        for sigma in scales:
            # Adjust sigma for spacing (anisotropy)
            sigma_pixel = [sigma / s for s in spacing]
            
            # --- Step 1: Calculate Hessian using Gaussian Derivatives ---
            # This is more accurate than Gaussian Blur + Finite Difference
            H_elems = hessian_matrix(self.image, sigma=sigma_pixel, order='rc', use_gaussian_derivatives=True)
            
            # --- Step 2: Eigenvalues ---
            # hessian_matrix_eigvals returns eigenvalues sorted by magnitude (abs value)
            # l1 is the smaller eigenvalue (along vessel), l2 is larger (across vessel)
            l1, l2 = hessian_matrix_eigvals(H_elems)
            
            # --- Step 3: Scale Normalization ---
            # Frangi requires multiplying derivatives by sigma^2 to be scale invariant
            norm_factor = sigma ** 2
            l1 *= norm_factor
            l2 *= norm_factor
            
            # --- Step 4: Frangi Vesselness Formula ---
            # Avoid div by zero
            l2[l2 == 0] = 1e-10
            
            RB = (l1 / l2) ** 2
            S2 = l1**2 + l2**2
            
            term_blob = np.exp(-RB / (2 * beta**2))
            term_noise = 1 - np.exp(-S2 / (2 * c_scaled**2))
            
            V = term_blob * term_noise
            
            # Filter negative responses (bright structures on dark background)
            # For bright vessels, the major curvature l2 should be negative
            V[l2 > 0] = 0
            
            # --- Step 5: Max Projection ---
            # Update max vesselness and scale map
            update_mask = V > max_vesselness
            max_vesselness[update_mask] = V[update_mask]
            scale_map[update_mask] = sigma
            
        return max_vesselness, scale_map
    
    def return_image(self):
        return self.image


# --- Usage Example ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import data

    # 1. Load Example Data (Retina image is good for vessels)
    # Using a 16-bit like range or just standard data
    raw_img = data.retina()[300:800, 300:800] # Crop for visibility
    
    # 2. Initialize Pipeline
    pipeline = ImageJProcessor(raw_img)
    
    # 3. Run Steps (You can comment out steps to remove them)
    pipeline.enhance_contrast_clahe(block_size=127, slope=3.0)
    pipeline.fft_bandpass_filter(large_structures_down_to=40, small_structures_up_to=3)
    pipeline.subtract_background_rolling_ball(radius=30, light_background=False)
    # pipeline.gaussian_blur(sigma=1.0) # Optional
    
    # 4. Run Frangi (Returns two images, does not overwrite pipeline state)
    vesselness, scale_map = pipeline.apply_frangi_filter(min_scale=1, max_scale=4, step_size=0.5)

    # 5. Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title('Original')
    
    # Pre-processed (Contrast + Bandpass + BG Sub)
    axes[1].imshow(pipeline.image, cmap='gray')
    axes[1].set_title('Pre-processed')
    
    # Frangi Vesselness
    axes[2].imshow(vesselness, cmap='magma')
    axes[2].set_title('Frangi Vesselness')
    
    # Frangi Scale Map
    im = axes[3].imshow(scale_map, cmap='jet')
    axes[3].set_title('Frangi Scale Map')
    plt.colorbar(im, ax=axes[3], label='Vessel Radius (px)')
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()