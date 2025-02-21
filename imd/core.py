"""
IMD Image Generation module
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.draw import disk
from skimage.restoration import rolling_ball
from skimage import measure
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.optimize import curve_fit
import cv2
from concurrent.futures import ThreadPoolExecutor
from skimage.restoration import rolling_ball
import napari

def create_imd(donor, acceptor, rmax=2.4, rmin=1.2, dmax=2000, dmin=10):
    """
    IMD (Intensity Modulated Display) image generation
    
    Parameters:
    -----------
    donor : numpy.ndarray
        Donor image stack (CFP)
    acceptor : numpy.ndarray
        Acceptor image stack (FRET)
    rmax, rmin : float
        Max (red) to Minimum (blue) ratio values
    dmax, dmin : float
        Donor fluorescence intensity Max and Minimus
    
    Returns:
    --------
    numpy.ndarray
        RGB IMD image stack
    """
    # Calculation of ranges
    rrange = rmax - rmin
    drange = dmax - dmin
    
    # Donor intensity-based mask images
    mask = donor.copy()
    mask = (mask - dmin) / drange
    mask = np.clip(mask, 0, 1)
    
    # Calculate ratio and scaling
    ratio = acceptor / donor
    ratio = (ratio - rmin) / rrange
    ratio = np.clip(ratio, 0, 1)
    ratio = (ratio * 255).astype(np.uint8)
    
    # physics colormap scaling (8 bit)
    colors = np.zeros((256, 3), dtype=np.uint8)
    # 赤成分
    colors[:, 0] = np.minimum(255, np.maximum(0, 255 * (2 - 4 * np.abs(np.linspace(0, 1, 256) - 0.75))))
    # 緑成分
    colors[:, 1] = np.minimum(255, np.maximum(0, 255 * (2 - 4 * np.abs(np.linspace(0, 1, 256) - 0.5))))
    # 青成分
    colors[:, 2] = np.minimum(255, np.maximum(0, 255 * (2 - 4 * np.abs(np.linspace(0, 1, 256) - 0.25))))
    
    # RGB channel generation
    rgb_stack = np.zeros((*ratio.shape, 3), dtype=np.uint8)
    for i in range(len(ratio)):
        # LUT
        rgb = colors[ratio[i]]
        # Mask
        for c in range(3):
            rgb_stack[i, :, :, c] = (rgb[:, :, c] * mask[i]).astype(np.uint8)
    
    return rgb_stack

def view_imd_with_napari(imd_stack, cfp_stack=None, fret_stack=None):
    """
    IMD, CFP, and FRET image stacks are visualized using napari
    
    Parameters:
    -----------
    imd_stack : numpy.ndarray
        RGB IMD image stack (frames, height, width, 3)
    cfp_stack : numpy.ndarray, optional
        CFP image stack
    fret_stack : numpy.ndarray, optional
        FRET image stack
    """
    # napari viewer format
    viewer = napari.Viewer()
    
    # add IMD images 
    viewer.add_image(
        imd_stack,
        name='IMD',
        rgb=True
    )
    
    # add CFP and FRET images (option)
    if cfp_stack is not None:
        viewer.add_image(
            cfp_stack,
            name='CFP',
            colormap='blue',
            visible=False  
        )
    
    if fret_stack is not None:
        viewer.add_image(
            fret_stack,
            name='FRET',
            colormap='green',
            visible=False 
        )
    
    return viewer

        
def rolling_ball_background_correction(img_stack, radius=50):
    """Rolling Ball method to subtract uneven background
        
    Parameters:
    -----------
    img_stack : np.ndarray
        Image stack
    radius : int
        radius of rolling ball
        
    Returns:
    --------
    np.ndarray
        Background-subtracted Image stack
    """
    
    if img_stack is None:
        print("Images are not loaded")
        return None
    
    corrected_stack = np.zeros_like(img_stack, dtype=np.float32)
    
    for i in range(img_stack.shape[0]):
        background = rolling_ball(img_stack[i], radius=radius)
        corrected_stack[i] = img_stack[i] - background
        
        if i % 10 == 0:
            print(f"Processing frame {i}/{img_stack.shape[0]}")
    
    return np.maximum(corrected_stack, 0)

def rolling_ball_background_correction_fast_with_options(img_stack, radius=50, scale_factor=1.0, use_gpu=False):
    """
    Improved version of Rolling Ball method to subtract uneven background.
    
    Parameters:
    -----------
    img_stack : np.ndarray
        Image stack
    radius : int
        radius of rolling ball
    scale_factor : float
        image scaling for fast clculation (smaller value than 1.0 reduces clculation time, but image might be blured)
    use_gpu : bool
        You can use GPU if your GPU is supported through OpenCV
        
    Returns:
    --------
    np.ndarray
        Background-subtracted Image stack
    """
    
    def process_single_frame(frame):
        # Save the original image size
        original_size = frame.shape
        
        # Scaling image（Option）
        if scale_factor != 1.0:
            new_size = tuple(int(dim * scale_factor) for dim in frame.shape)
            frame = cv2.resize(frame, (new_size[1], new_size[0]))
        
        # Calculate kernel size
        kernel_size = int(2 * radius * scale_factor) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if use_gpu:
            # Transfer data to GPU memory
            frame_gpu = cv2.UMat(frame)
            background_gpu = cv2.morphologyEx(frame_gpu, cv2.MORPH_OPEN, kernel)
            corrected_gpu = cv2.subtract(frame_gpu, background_gpu)
            corrected = cv2.UMat.get(corrected_gpu)
        else:
            background = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            corrected = frame - background
        
        # Resize (Option)
        if scale_factor != 1.0:
            corrected = cv2.resize(corrected, (original_size[1], original_size[0]))
        
        return np.maximum(corrected, 0)
    
    # Convert image 
    img_stack = img_stack.astype(np.float32)
    
    # Parallel processing
    with ThreadPoolExecutor() as executor:
        corrected_stack = list(executor.map(process_single_frame, img_stack))
    
    return np.array(corrected_stack)
    
def compare_background_correction(original, corrected, frame_idx=0):
    """Comparison of original and background-subtracted image"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    vmin1 = np.percentile(original[frame_idx], 1)
    vmax1 = np.percentile(original[frame_idx], 99)
    im1 = ax1.imshow(original[frame_idx], cmap='gray', vmin=vmin1, vmax=vmax1)
    ax1.set_title('Original')
    plt.colorbar(im1, ax=ax1)
    ax1.axis('off')
    
    vmin2 = np.percentile(corrected[frame_idx], 1)
    vmax2 = np.percentile(corrected[frame_idx], 99)
    im2 = ax2.imshow(corrected[frame_idx], cmap='gray', vmin=vmin2, vmax=vmax2)
    ax2.set_title('Background Corrected')
    plt.colorbar(im2, ax=ax2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    