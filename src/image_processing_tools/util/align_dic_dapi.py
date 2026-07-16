import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel
from image_processing_tools.image_class.image_filters import ImageJProcessor
from image_processing_tools.image_class.image_container import ImageContainer

# Requires too much manual tuning of filters. Abandoned.

def calculate_dic_shift(ref_img_container, dic_img_container):
    
    ref_img = ref_img_container.merge()
    dic_img = dic_img_container.merge()
    
    ref_img = ref_img_container.merge()
    ref_img_filters = ImageJProcessor(ref_img)
    ref_img_filters.fft_bandpass_filter()
    ref_img_filters.frangi_imagej_ops()
    ref_img = ref_img_filters.return_image()

    mov_img = dic_img_container.merge()
    mov_img_filters = ImageJProcessor(mov_img)
    mov_img_filters.enhance_contrast_clahe()
    mov_img_filters.fft_bandpass_filter()
    mov_img_filters.imagej_rolling_ball()
    mov_img = mov_img_filters.return_image()
    
    ref_edges = sobel(ref_img.astype(np.float32) / ref_img.max())
    mov_edges = sobel(dic_img.astype(np.float32) / dic_img.max())
    
    shift, _, _ = phase_cross_correlation(
        ref_edges, 
        mov_edges, 
        upsample_factor=10  # Sub-pixel accuracy down to 1/10th of a pixel
    )
    
    # 4. Convert to integers since scipy.ndimage.shift expects whole numbers
    rounded_shift = [int(round(shift[0])), int(round(shift[1]))]
    
    print(f"Calculated shift for 'DIC image': {rounded_shift}")
    return rounded_shift
    