#!/usr/bin/env python
"""
Debug Farneback algorithm step by step.

This script runs the Farneback algorithm with detailed debugging output
at each step to identify potential bugs.
"""

import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from tqdm import tqdm

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the Farneback implementation
from Farneback_Numba_optimized import Farneback_Numba

def load_parabolic_images():
    """Load the parabolic test images from the Bits08 dataset."""
    base_path = os.path.join('testImages', 'Bits08', 'Ni06')
    fn1 = os.path.join(base_path, 'parabolic01_0.tif')
    fn2 = os.path.join(base_path, 'parabolic01_1.tif')

    print(f"Loading images from {fn1} and {fn2}")

    # Load images
    im1 = imread(fn1).astype(np.float32)
    im2 = imread(fn2).astype(np.float32)

    return im1, im2

def visualize_image(img, title, output_file=None):
    """Visualize an image with matplotlib."""
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.colorbar(label='Intensity')

    if output_file:
        plt.savefig(output_file, dpi=200)
        print(f"Image saved to {output_file}")

    plt.close()

def visualize_flow(U, V, title, output_file=None):
    """Visualize flow field as quiver plot."""
    # Create figure
    plt.figure(figsize=(12, 10))

    # Create quiver plot with color-coded arrows based on magnitude
    quiver_skip = 20
    y, x = np.mgrid[0:U.shape[0]:quiver_skip, 0:U.shape[1]:quiver_skip]
    u_skip = U[::quiver_skip, ::quiver_skip]
    v_skip = V[::quiver_skip, ::quiver_skip]

    # Calculate magnitude for coloring
    magnitude = np.sqrt(u_skip**2 + v_skip**2)

    # Plot quiver with colors based on magnitude
    quiv = plt.quiver(x, y, u_skip, v_skip, magnitude,
                     scale=50, scale_units='inches',
                     cmap='jet', clim=[0, np.percentile(magnitude, 95)])
    plt.colorbar(quiv, label='Magnitude (pixels/frame)')

    plt.title(title)
    plt.xlim(0, U.shape[1])
    plt.ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file, dpi=200)
        print(f"Flow visualization saved to {output_file}")

    plt.close()

def visualize_horizontal_profile(U, V, title, output_file=None):
    """Visualize horizontal profile of flow field."""
    # Extract middle row
    position = U.shape[0] // 2

    x = np.arange(U.shape[1])
    u = U[position, :]
    v = V[position, :]

    # Create figure with two subplots
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(x, u, 'b-', linewidth=2)
    plt.title(f'{title} - Horizontal Velocity Profile')
    plt.xlabel('X (pixels)')
    plt.ylabel('Horizontal Velocity (pixels/frame)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x, v, 'r-', linewidth=2)
    plt.title(f'{title} - Vertical Velocity Profile')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=200)
        print(f"Profile visualization saved to {output_file}")

    plt.close()

def debug_polynomial_expansion(fb, img, output_dir):
    """Debug polynomial expansion step."""
    print("\nDebugging polynomial expansion...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Visualize input image
    visualize_image(img, "Input Image", os.path.join(output_dir, "input_image.png"))

    # Initialize R matrix
    R = np.zeros((5*img.shape[0], img.shape[1]), np.float32)

    # Run polynomial expansion
    try:
        start_time = time.time()
        R = fb.polynomialExpansion(img, R)
        elapsed_time = time.time() - start_time
        print(f"Polynomial expansion completed in {elapsed_time:.2f} seconds")

        # Extract and visualize polynomial coefficients
        height = img.shape[0]
        for k in range(5):
            coef = R[k::5, :]  # Extract every 5th row starting from k
            visualize_image(coef, f"Polynomial Coefficient {k+1}",
                          os.path.join(output_dir, f"poly_coef_{k+1}.png"))

            # Print statistics
            print(f"Coefficient {k+1}: min={coef.min():.4f}, max={coef.max():.4f}, mean={coef.mean():.4f}")

        return R
    except Exception as e:
        print(f"Error in polynomial expansion: {e}")
        return None

def debug_update_matrices(fb, flowX, flowY, RA, RB, output_dir):
    """Debug update matrices step."""
    print("\nDebugging update matrices...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Visualize input flow
    visualize_flow(flowX, flowY, "Input Flow", os.path.join(output_dir, "input_flow.png"))

    # Initialize M matrix
    M = np.zeros_like(RA)

    # Run update matrices
    try:
        start_time = time.time()
        M = fb.updateMatrices(flowX, flowY, RA, RB, M)
        elapsed_time = time.time() - start_time
        print(f"Update matrices completed in {elapsed_time:.2f} seconds")

        # Extract and visualize M coefficients
        height = flowX.shape[0]
        for k in range(5):
            coef = M[k::5, :]  # Extract every 5th row starting from k
            visualize_image(coef, f"M Coefficient {k+1}",
                          os.path.join(output_dir, f"M_coef_{k+1}.png"))

            # Print statistics
            print(f"M Coefficient {k+1}: min={coef.min():.4f}, max={coef.max():.4f}, mean={coef.mean():.4f}")

        return M
    except Exception as e:
        print(f"Error in update matrices: {e}")
        return None

def debug_update_flow(fb, M, flowX, flowY, output_dir):
    """Debug update flow step."""
    print("\nDebugging update flow...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Visualize input flow
    visualize_flow(flowX, flowY, "Input Flow", os.path.join(output_dir, "input_flow.png"))

    # Run update flow
    try:
        start_time = time.time()
        flowX_new, flowY_new = fb.updateFlow(M, flowX.copy(), flowY.copy())
        elapsed_time = time.time() - start_time
        print(f"Update flow completed in {elapsed_time:.2f} seconds")

        # Visualize updated flow
        visualize_flow(flowX_new, flowY_new, "Updated Flow", os.path.join(output_dir, "updated_flow.png"))

        # Visualize flow difference
        flow_diff_x = flowX_new - flowX
        flow_diff_y = flowY_new - flowY
        visualize_flow(flow_diff_x, flow_diff_y, "Flow Difference", os.path.join(output_dir, "flow_diff.png"))

        # Print statistics
        print(f"Original flow: X min={flowX.min():.4f}, max={flowX.max():.4f}, Y min={flowY.min():.4f}, max={flowY.max():.4f}")
        print(f"Updated flow: X min={flowX_new.min():.4f}, max={flowX_new.max():.4f}, Y min={flowY_new.min():.4f}, max={flowY_new.max():.4f}")
        print(f"Flow difference: X min={flow_diff_x.min():.4f}, max={flow_diff_x.max():.4f}, Y min={flow_diff_y.min():.4f}, max={flow_diff_y.max():.4f}")

        return flowX_new, flowY_new
    except Exception as e:
        print(f"Error in update flow: {e}")
        return flowX, flowY

def debug_gaussian_blur(fb, M, output_dir):
    """Debug Gaussian blur step."""
    print("\nDebugging Gaussian blur...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize bufM
    bufM = np.zeros_like(M)

    # Run Gaussian blur
    try:
        start_time = time.time()
        bufM = fb.gaussianBlur5(M, fb.windowSize//2, bufM)
        elapsed_time = time.time() - start_time
        print(f"Gaussian blur completed in {elapsed_time:.2f} seconds")

        # Extract and visualize blurred coefficients
        height = M.shape[0] // 5
        for k in range(5):
            coef_orig = M[k::5, :]  # Extract every 5th row starting from k
            coef_blur = bufM[k::5, :]

            # Visualize original and blurred coefficients
            plt.figure(figsize=(16, 8))

            plt.subplot(1, 2, 1)
            plt.imshow(coef_orig, cmap='viridis')
            plt.title(f"Original M Coefficient {k+1}")
            plt.colorbar(label='Value')

            plt.subplot(1, 2, 2)
            plt.imshow(coef_blur, cmap='viridis')
            plt.title(f"Blurred M Coefficient {k+1}")
            plt.colorbar(label='Value')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"blur_coef_{k+1}.png"), dpi=200)
            plt.close()

            # Print statistics
            print(f"Original Coefficient {k+1}: min={coef_orig.min():.4f}, max={coef_orig.max():.4f}, mean={coef_orig.mean():.4f}")
            print(f"Blurred Coefficient {k+1}: min={coef_blur.min():.4f}, max={coef_blur.max():.4f}, mean={coef_blur.mean():.4f}")

        return bufM
    except Exception as e:
        print(f"Error in Gaussian blur: {e}")
        return None

def debug_farneback_iteration(fb, RA, RB, flowX, flowY, M, bufM, iteration, output_dir):
    """Debug a single Farneback iteration."""
    print(f"\nDebugging Farneback iteration {iteration}...")

    # Create output directory
    iter_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    # Visualize current flow
    visualize_flow(flowX, flowY, f"Flow at Iteration {iteration}",
                 os.path.join(iter_dir, "flow.png"))
    visualize_horizontal_profile(flowX, flowY, f"Flow at Iteration {iteration}",
                               os.path.join(iter_dir, "profile.png"))

    # Apply Gaussian blur
    try:
        bufM = debug_gaussian_blur(fb, M, os.path.join(iter_dir, "blur"))
        if bufM is None:
            print("Gaussian blur failed, skipping rest of iteration")
            return flowX, flowY, M, bufM

        # Swap M and bufM
        M, bufM = bufM, M

        # Update flow
        flowX, flowY = debug_update_flow(fb, M, flowX, flowY, os.path.join(iter_dir, "update_flow"))

        # Update matrices for next iteration
        if iteration < fb.numIters - 1:
            M = debug_update_matrices(fb, flowX, flowY, RA, RB, os.path.join(iter_dir, "update_matrices"))
            if M is None:
                print("Update matrices failed, skipping rest of iteration")
                return flowX, flowY, M, bufM

        return flowX, flowY, M, bufM
    except Exception as e:
        print(f"Error in iteration {iteration}: {e}")
        return flowX, flowY, M, bufM

def debug_farneback_level(fb, im1, im2, flowX, flowY, level, scale, output_dir):
    """Debug a single pyramid level of Farneback algorithm."""
    print(f"\nDebugging Farneback pyramid level {level} (scale={scale:.3f})...")

    # Create output directory
    level_dir = os.path.join(output_dir, f"level_{level}")
    os.makedirs(level_dir, exist_ok=True)

    # Resize images and flow for current level
    height, width = im1.shape
    level_height = int(round(height * scale))
    level_width = int(round(width * scale))

    # Prepare images for current level
    sigma = (1.0/scale - 1.0) * 0.5
    smoothSize = int(round(sigma*5)) | 1
    smoothSize = max(smoothSize, 3)

    print(f"Level {level} image size: {level_height}x{level_width}, scale: {scale:.3f}, sigma: {sigma:.3f}")

    # Apply Gaussian blur and resize
    from scipy.ndimage import gaussian_filter
    blurredFrameA = gaussian_filter(im1, sigma) if sigma > 0 else im1
    blurredFrameB = gaussian_filter(im2, sigma) if sigma > 0 else im2

    # Resize images using PIL
    from PIL import Image
    pyrLevelA = np.array(Image.fromarray(blurredFrameA).resize((level_width, level_height), Image.BILINEAR))
    pyrLevelB = np.array(Image.fromarray(blurredFrameB).resize((level_width, level_height), Image.BILINEAR))

    # Visualize pyramid level images
    visualize_image(pyrLevelA, f"Pyramid Level {level} - Image A",
                  os.path.join(level_dir, "image_a.png"))
    visualize_image(pyrLevelB, f"Pyramid Level {level} - Image B",
                  os.path.join(level_dir, "image_b.png"))

    # Resize flow for current level
    if flowX.shape != (level_height, level_width):
        # Resize flow using PIL
        from PIL import Image
        flowX_resized = np.array(Image.fromarray(flowX).resize((level_width, level_height), Image.BILINEAR))
        flowY_resized = np.array(Image.fromarray(flowY).resize((level_width, level_height), Image.BILINEAR))

        # Scale flow values
        if level > 0:  # Not the first level
            flowX_resized *= 1.0/fb.pyrScale
            flowY_resized *= 1.0/fb.pyrScale

        flowX = flowX_resized
        flowY = flowY_resized

    # Visualize initial flow for this level
    visualize_flow(flowX, flowY, f"Initial Flow for Level {level}",
                 os.path.join(level_dir, "initial_flow.png"))

    # Initialize matrices
    M = np.zeros((5*level_height, level_width), np.float32)
    bufM = np.zeros((5*level_height, level_width), np.float32)
    RA = np.zeros((5*level_height, level_width), np.float32)
    RB = np.zeros((5*level_height, level_width), np.float32)

    # Polynomial expansion
    RA = debug_polynomial_expansion(fb, pyrLevelA, os.path.join(level_dir, "poly_exp_a"))
    if RA is None:
        print("Polynomial expansion for frame A failed, skipping level")
        return flowX, flowY

    RB = debug_polynomial_expansion(fb, pyrLevelB, os.path.join(level_dir, "poly_exp_b"))
    if RB is None:
        print("Polynomial expansion for frame B failed, skipping level")
        return flowX, flowY

    # Update matrices based on current flow
    M = debug_update_matrices(fb, flowX, flowY, RA, RB, os.path.join(level_dir, "initial_matrices"))
    if M is None:
        print("Initial update matrices failed, skipping level")
        return flowX, flowY

    # Iteratively update flow
    for i in range(fb.numIters):
        flowX, flowY, M, bufM = debug_farneback_iteration(
            fb, RA, RB, flowX, flowY, M, bufM, i, os.path.join(level_dir, "iterations"))

    # Visualize final flow for this level
    visualize_flow(flowX, flowY, f"Final Flow for Level {level}",
                 os.path.join(level_dir, "final_flow.png"))
    visualize_horizontal_profile(flowX, flowY, f"Final Flow for Level {level}",
                               os.path.join(level_dir, "final_profile.png"))

    return flowX, flowY

def debug_farneback_algorithm(im1, im2, window_size, iterations=5, poly_n=5, poly_sigma=1.2, pyr_levels=1, output_dir="farneback_debug"):
    """
    Debug Farneback algorithm step by step.

    Args:
        im1, im2: Input images
        window_size: Window size for Farneback
        iterations: Number of iterations
        poly_n: Polynomial degree
        poly_sigma: Polynomial sigma
        pyr_levels: Number of pyramid levels
        output_dir: Output directory for debug visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Create Farneback adapter
    fb = Farneback_Numba(
        windowSize=window_size,
        Niters=iterations,
        polyN=poly_n,
        polySigma=poly_sigma,
        pyramidalLevels=pyr_levels
    )

    # Print parameters
    print(f"\nDebugging Farneback algorithm with parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Polynomial degree: {poly_n}")
    print(f"  Polynomial sigma: {poly_sigma}")
    print(f"  Pyramid levels: {pyr_levels}")

    # Visualize input images
    visualize_image(im1, "Input Image 1", os.path.join(output_dir, "input_image1.png"))
    visualize_image(im2, "Input Image 2", os.path.join(output_dir, "input_image2.png"))

    # Debug each pyramid level
    finalNumLevels = fb.pyramidalLevels
    size = im1.shape

    for k in range(finalNumLevels, -1, -1):
        scale = 1.0
        for i in range(k):
            scale *= fb.pyrScale

        U, V = debug_farneback_level(fb, im1, im2, U, V, k, scale, output_dir)

    # Visualize final flow
    visualize_flow(U, V, f"Final Flow (window={window_size})",
                 os.path.join(output_dir, "final_flow.png"))
    visualize_horizontal_profile(U, V, f"Final Flow (window={window_size})",
                               os.path.join(output_dir, "final_profile.png"))

    # Save flow field
    results = {
        'U': U,
        'V': V,
        'window_size': np.array([[window_size]]),
        'iterations': np.array([[iterations]]),
        'poly_n': np.array([[poly_n]]),
        'poly_sigma': np.array([[poly_sigma]]),
        'pyr_levels': np.array([[pyr_levels]])
    }

    output_file = os.path.join(output_dir, f'farneback_window{window_size}.mat')
    savemat(output_file, results)
    print(f"Flow saved to {output_file}")

    return U, V

def main():
    # Load test images
    im1, im2 = load_parabolic_images()

    # Debug Farneback algorithm with window size 15
    window_size = 15
    iterations = 5
    poly_n = 5
    poly_sigma = 1.2
    pyr_levels = 1

    output_dir = f'farneback_debug_window{window_size}'

    U, V = debug_farneback_algorithm(
        im1, im2,
        window_size=window_size,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        pyr_levels=pyr_levels,
        output_dir=output_dir
    )

    print("\nDebug complete. Results saved to", output_dir)

if __name__ == "__main__":
    main()
