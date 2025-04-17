#!/usr/bin/env python
"""
Run Horn-Schunck optical flow algorithm with Liu-Shen enhancement on 17apr image pair
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.io import imread
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from GenericPyramidalOpticalFlowWrapper import GenericPyramidalOpticalFlowWrapper
from PhysicsBasedOpticalFlowLiuShen import LiuShenOpticalFlowAlgoAdapter
from HornSchunck import HSOpticalFlowAlgoAdapter

def save_flow(U, V, filename):
    margins = {
        'top': 0,
        'left': 0,
        'bottom': 0,
        'right': 0
    }
    results = {
        'u': U,
        'v': V,
        'iaWidth': 1,
        'iaHeight': 1,
        'margins': margins
    }

    parameters = {
        'overlapFactor': 1.0,
        'imageHeight': np.size(U, 0),
        'imageWidth': np.size(U, 1)
    }

    scipy.io.savemat(filename, mdict={'velocities': results, 'parameters': parameters})

def plot_results(U, V, title, output_filename, vmin=None, vmax=None, quiver_scale=50, quiver_skip=40):
    """
    Plot optical flow results as colormesh and quiver plots

    Args:
        U, V: Horizontal and vertical velocity components
        title: Plot title
        output_filename: Filename for saving plot
        vmin, vmax: Color scale limits for vertical velocity (auto-calculated if None)
        quiver_scale: Scale factor for quiver plot
        quiver_skip: Skip factor for quiver plot (to avoid overcrowding)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Auto-calculate color limits if not provided
    if vmin is None or vmax is None:
        # Calculate percentiles to exclude outliers
        v_abs_max = max(abs(np.percentile(V, 1)), abs(np.percentile(V, 99)))
        vmin = -v_abs_max
        vmax = v_abs_max

    # Plot vertical velocity component as colormesh with high contrast colormap
    im1 = ax1.imshow(V, cmap='jet', norm=Normalize(vmin=vmin, vmax=vmax))
    ax1.set_title(f'{title} - Vertical Velocity (v)')
    plt.colorbar(im1, ax=ax1, label='Pixels/frame')

    # Add a grid for better reference
    ax1.grid(False)

    # Create quiver plot with color-coded arrows based on magnitude
    y, x = np.mgrid[0:U.shape[0]:quiver_skip, 0:U.shape[1]:quiver_skip]
    u_skip = U[::quiver_skip, ::quiver_skip]
    v_skip = V[::quiver_skip, ::quiver_skip]

    # Calculate magnitude for coloring
    magnitude = np.sqrt(u_skip**2 + v_skip**2)

    # Plot quiver with colors based on magnitude
    quiv = ax2.quiver(x, y, u_skip, v_skip, magnitude,
                     scale=quiver_scale, scale_units='inches',
                     cmap='jet', clim=[0, np.percentile(magnitude, 95)])
    plt.colorbar(quiv, ax=ax2, label='Magnitude (pixels/frame)')

    ax2.set_title(f'{title} - Vector Field')
    ax2.set_xlim(0, U.shape[1])
    ax2.set_ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

def main():
    # Define image paths
    img1_path = "examples/testImages/17apr/20-37-11.000-6.tif"
    img2_path = "examples/testImages/17apr/20-37-11.000-8.tif"

    # Load images
    print(f"Loading images: {img1_path} and {img2_path}")
    img1 = imread(img1_path).astype(np.float32)
    img2 = imread(img2_path).astype(np.float32)

    # Print image information
    print(f"Image shape: {img1.shape}")
    print(f"Image dtype: {img1.dtype}")
    print(f"Image min/max values: {img1.min()}/{img1.max()}")

    # Normalize images if needed (16-bit to 8-bit range)
    if img1.max() > 255:
        img1 = (img1 / 65535.0 * 255.0).astype(np.float32)
        img2 = (img2 / 65535.0 * 255.0).astype(np.float32)
        print("Images normalized from 16-bit to 8-bit range")

    # Create output directory if it doesn't exist
    output_dir = 'hornschunck_liushen_17apr_results'
    os.makedirs(output_dir, exist_ok=True)

    # Define Horn-Schunck parameters
    alpha = 1.0
    iterations = 100
    pyr_levels = 2

    # Run Horn-Schunck without Liu-Shen enhancement first
    print("\nRunning Horn-Schunck without Liu-Shen enhancement")

    # Create Horn-Schunck adapter with enough alphas for all iterations
    # Each pyramidal level and k-level iteration will pop one alpha
    num_alphas = pyr_levels * 1  # 1 k-level
    hs_adapter = HSOpticalFlowAlgoAdapter(alphas=[alpha] * num_alphas, Niter=iterations, provideGenericPyramidalDefaults=True)

    # Create pyramidal optical flow wrapper
    pyr_of = GenericPyramidalOpticalFlowWrapper(
        hs_adapter,
        filter_sigma=3.4,
        pyr_levels=pyr_levels
    )

    # Run the algorithm
    import time
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "HornSchunck_PyrLvls2")
    save_flow(U, V, f"{output_base}.mat")

    # Plot results
    plot_results(U, V, "Horn-Schunck", f"{output_base}.png")

    # Run Horn-Schunck with Liu-Shen enhancement
    print("\nRunning Horn-Schunck with Liu-Shen enhancement")

    # Create Horn-Schunck adapter with enough alphas for all iterations
    num_alphas = pyr_levels * 1  # 1 k-level
    hs_adapter = HSOpticalFlowAlgoAdapter(alphas=[alpha] * num_alphas, Niter=iterations, provideGenericPyramidalDefaults=True)

    # Create Liu-Shen adapter
    ls_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)  # Alpha parameter for regularization

    # Create pyramidal optical flow wrapper
    pyr_of = GenericPyramidalOpticalFlowWrapper(
        hs_adapter,
        filter_sigma=3.4,
        pyr_levels=pyr_levels,
        optional_algo_adapter=ls_adapter,
        filter_opt=0.48  # Add filter_opt parameter for Liu-Shen
    )

    # Run the algorithm
    start_time = time.time()
    U_ls, V_ls = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U_ls.min():.2f} to {U_ls.max():.2f}")
    print(f"Flow range V: {V_ls.min():.2f} to {V_ls.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "HornSchunck_LiuShen_PyrLvls2")
    save_flow(U_ls, V_ls, f"{output_base}.mat")

    # Plot results
    plot_results(U_ls, V_ls, "Horn-Schunck with Liu-Shen Enhancement", f"{output_base}.png")

    # Plot difference between Liu-Shen and regular Horn-Schunck
    diff_U = U_ls - U
    diff_V = V_ls - V

    output_base = os.path.join(output_dir, "HornSchunck_LiuShen_Difference")
    plot_results(diff_U, diff_V, "Difference (Liu-Shen - Regular)", f"{output_base}.png")

    print(f"\nResults saved to {output_dir} directory")

if __name__ == "__main__":
    main()
