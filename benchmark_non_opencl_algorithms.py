#!/usr/bin/env python
"""
Benchmark script for non-OpenCL optical flow algorithms on the Bits08 test case.

This script compares the following algorithms:
1. Farneback_Numba
2. denseLucasKanade_Numba
3. HornSchunck

It analyzes both the flow fields and parabolic profiles.
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
from scipy.ndimage import gaussian_filter
import scipy.io
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Import algorithms
from Farneback_Numba import Farneback_Numba
from denseLucasKanade_Numba import denseLucasKanade_Numba
from HornSchunck import HSOpticalFlowAlgoAdapter
from GenericPyramidalOpticalFlowWrapper import GenericPyramidalOpticalFlowWrapper
from PhysicsBasedOpticalFlowLiuShen import LiuShenOpticalFlowAlgoAdapter

def load_bits08_images():
    """Load the Bits08 test case images."""
    img1_path = "examples/testImages/Bits08/Ni06/parabolic01_0.tif"
    img2_path = "examples/testImages/Bits08/Ni06/parabolic01_1.tif"

    print(f"Loading images: {img1_path} and {img2_path}")
    with tqdm(total=100, desc="Loading first image", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        img1 = imread(img1_path).astype(np.float32)
        pbar.update(100)

    with tqdm(total=100, desc="Loading second image", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        img2 = imread(img2_path).astype(np.float32)
        pbar.update(100)

    # Print image information
    print(f"Image shape: {img1.shape}")
    print(f"Image dtype: {img1.dtype}")
    print(f"Image min/max values: {img1.min()}/{img1.max()}")

    # Normalize images if needed (16-bit to 8-bit range)
    if img1.max() > 255:
        with tqdm(total=100, desc="Normalizing images", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            img1 = (img1 / 65535.0 * 255.0).astype(np.float32)
            img2 = (img2 / 65535.0 * 255.0).astype(np.float32)
            print("Images normalized from 16-bit to 8-bit range")
            pbar.update(100)

    return img1, img2

def save_flow(U, V, filename):
    """Save flow fields to a .mat file."""
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

def plot_flow_field(U, V, title, output_filename, vmin=None, vmax=None, quiver_scale=50, quiver_skip=20):
    """
    Plot optical flow as colormesh and quiver plots.

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

def extract_parabolic_profile(U, V, axis='horizontal', position=None):
    """
    Extract a parabolic profile from the flow field.

    Args:
        U, V: Horizontal and vertical velocity components
        axis: 'horizontal' or 'vertical'
        position: Position along the perpendicular axis (default: middle)

    Returns:
        x: Position along the axis
        v: Velocity profile
    """
    if axis == 'horizontal':
        if position is None:
            position = U.shape[0] // 2
        x = np.arange(U.shape[1])
        v = V[position, :]
    else:  # vertical
        if position is None:
            position = U.shape[1] // 2
        x = np.arange(U.shape[0])
        v = V[:, position]

    return x, v

def plot_parabolic_profiles(profiles, title, output_filename):
    """
    Plot parabolic profiles from different algorithms.

    Args:
        profiles: Dictionary of profiles {algorithm_name: (x, v)}
        title: Plot title
        output_filename: Filename for saving plot
    """
    plt.figure(figsize=(12, 8))

    for algo_name, (x, v) in profiles.items():
        plt.plot(x, v, label=algo_name)

    plt.title(title)
    plt.xlabel('Position (pixels)')
    plt.ylabel('Velocity (pixels/frame)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

def run_farneback(img1, img2, output_dir):
    """Run Farneback algorithm and return flow fields."""
    print("\nRunning Farneback (Numba) algorithm...")

    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    # Create Farneback adapter with progress bar
    with tqdm(total=100, desc="Setting up Farneback", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        fb_adapter = Farneback_Numba(
            windowSize=33,
            Niters=5,
            polyN=7,
            polySigma=1.5,
            pyramidalLevels=3
        )
        pbar.update(100)

    # Create pyramidal optical flow wrapper
    with tqdm(total=100, desc="Creating pyramid wrapper", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pyr_of = GenericPyramidalOpticalFlowWrapper(
            fb_adapter,
            filter_sigma=0.0,
            pyr_levels=3
        )
        pbar.update(100)

    # Run the algorithm
    print("Computing Farneback optical flow...")
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "Farneback_Numba")
    save_flow(U, V, f"{output_base}.mat")

    # Plot results
    plot_flow_field(U, V, "Farneback (Numba)", f"{output_base}.png")

    return U, V, elapsed_time

def run_farneback_liushen(img1, img2, output_dir):
    """Run Farneback algorithm with Liu-Shen enhancement and return flow fields."""
    print("\nRunning Farneback (Numba) with Liu-Shen enhancement...")

    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    # Create Farneback adapter with progress bar
    with tqdm(total=100, desc="Setting up Farneback", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        fb_adapter = Farneback_Numba(
            windowSize=33,
            Niters=5,
            polyN=7,
            polySigma=1.5,
            pyramidalLevels=3
        )
        pbar.update(100)

    # Create Liu-Shen adapter
    with tqdm(total=100, desc="Setting up Liu-Shen", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        ls_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)  # Alpha parameter for regularization
        pbar.update(100)

    # Create pyramidal optical flow wrapper
    with tqdm(total=100, desc="Creating pyramid wrapper", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pyr_of = GenericPyramidalOpticalFlowWrapper(
            fb_adapter,
            filter_sigma=0.0,
            pyr_levels=3,
            optional_algo_adapter=ls_adapter,
            filter_opt=0.48  # Add filter_opt parameter for Liu-Shen
        )
        pbar.update(100)

    # Run the algorithm
    print("Computing Farneback with Liu-Shen optical flow...")
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "Farneback_Numba_LiuShen")
    save_flow(U, V, f"{output_base}.mat")

    # Plot results
    plot_flow_field(U, V, "Farneback (Numba) with Liu-Shen", f"{output_base}.png")

    return U, V, elapsed_time

def run_dense_lucas_kanade(img1, img2, output_dir):
    """Run dense Lucas-Kanade algorithm and return flow fields."""
    print("\nRunning Dense Lucas-Kanade (Numba) algorithm...")

    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    # Create dense Lucas-Kanade adapter with progress bar
    with tqdm(total=100, desc="Setting up Dense Lucas-Kanade", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        lk_adapter = denseLucasKanade_Numba(
            Niter=5,
            halfWindow=13,
            provideGenericPyramidalDefaults=True
        )
        pbar.update(100)

    # Create pyramidal optical flow wrapper
    with tqdm(total=100, desc="Creating pyramid wrapper", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pyr_of = GenericPyramidalOpticalFlowWrapper(
            lk_adapter,
            filter_sigma=0.0,
            pyr_levels=3
        )
        pbar.update(100)

    # Run the algorithm
    print("Computing Dense Lucas-Kanade optical flow...")
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "DenseLucasKanade_Numba")
    save_flow(U, V, f"{output_base}.mat")

    # Plot results
    plot_flow_field(U, V, "Dense Lucas-Kanade (Numba)", f"{output_base}.png")

    return U, V, elapsed_time

def run_dense_lucas_kanade_liushen(img1, img2, output_dir):
    """Run dense Lucas-Kanade algorithm with Liu-Shen enhancement and return flow fields."""
    print("\nRunning Dense Lucas-Kanade (Numba) with Liu-Shen enhancement...")

    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    # Create dense Lucas-Kanade adapter with progress bar
    with tqdm(total=100, desc="Setting up Dense Lucas-Kanade", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        lk_adapter = denseLucasKanade_Numba(
            Niter=5,
            halfWindow=13,
            provideGenericPyramidalDefaults=True
        )
        pbar.update(100)

    # Create Liu-Shen adapter
    with tqdm(total=100, desc="Setting up Liu-Shen", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        ls_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)  # Alpha parameter for regularization
        pbar.update(100)

    # Create pyramidal optical flow wrapper
    with tqdm(total=100, desc="Creating pyramid wrapper", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pyr_of = GenericPyramidalOpticalFlowWrapper(
            lk_adapter,
            filter_sigma=0.0,
            pyr_levels=3,
            optional_algo_adapter=ls_adapter,
            filter_opt=0.48  # Add filter_opt parameter for Liu-Shen
        )
        pbar.update(100)

    # Run the algorithm
    print("Computing Dense Lucas-Kanade with Liu-Shen optical flow...")
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "DenseLucasKanade_Numba_LiuShen")
    save_flow(U, V, f"{output_base}.mat")

    # Plot results
    plot_flow_field(U, V, "Dense Lucas-Kanade (Numba) with Liu-Shen", f"{output_base}.png")

    return U, V, elapsed_time

def run_horn_schunck(img1, img2, output_dir):
    """Run Horn-Schunck algorithm and return flow fields."""
    print("\nRunning Horn-Schunck algorithm...")

    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    # Create Horn-Schunck adapter with progress bar
    with tqdm(total=100, desc="Setting up Horn-Schunck", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        hs_adapter = HSOpticalFlowAlgoAdapter(
            alphas=[0.1],  # List of alpha values (one per pyramid level)
            Niter=100,
            provideGenericPyramidalDefaults=True
        )
        pbar.update(100)

    # Create pyramidal optical flow wrapper
    with tqdm(total=100, desc="Creating pyramid wrapper", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pyr_of = GenericPyramidalOpticalFlowWrapper(
            hs_adapter,
            filter_sigma=0.0,
            pyr_levels=3
        )
        pbar.update(100)

    # Run the algorithm
    print("Computing Horn-Schunck optical flow...")
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "HornSchunck")
    save_flow(U, V, f"{output_base}.mat")

    # Plot results
    plot_flow_field(U, V, "Horn-Schunck", f"{output_base}.png")

    return U, V, elapsed_time

def run_horn_schunck_liushen(img1, img2, output_dir):
    """Run Horn-Schunck algorithm with Liu-Shen enhancement and return flow fields."""
    print("\nRunning Horn-Schunck with Liu-Shen enhancement...")

    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)

    # Create Horn-Schunck adapter with progress bar
    with tqdm(total=100, desc="Setting up Horn-Schunck", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        hs_adapter = HSOpticalFlowAlgoAdapter(
            alphas=[0.1],  # List of alpha values (one per pyramid level)
            Niter=100,
            provideGenericPyramidalDefaults=True
        )
        pbar.update(100)

    # Create Liu-Shen adapter
    with tqdm(total=100, desc="Setting up Liu-Shen", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        ls_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)  # Alpha parameter for regularization
        pbar.update(100)

    # Create pyramidal optical flow wrapper
    with tqdm(total=100, desc="Creating pyramid wrapper", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pyr_of = GenericPyramidalOpticalFlowWrapper(
            hs_adapter,
            filter_sigma=0.0,
            pyr_levels=3,
            optional_algo_adapter=ls_adapter,
            filter_opt=0.48  # Add filter_opt parameter for Liu-Shen
        )
        pbar.update(100)

    # Run the algorithm
    print("Computing Horn-Schunck with Liu-Shen optical flow...")
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time

    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Save flow data
    output_base = os.path.join(output_dir, "HornSchunck_LiuShen")
    save_flow(U, V, f"{output_base}.mat")

    # Plot results
    plot_flow_field(U, V, "Horn-Schunck with Liu-Shen", f"{output_base}.png")

    return U, V, elapsed_time

def compare_all_algorithms(horizontal_profiles, vertical_profiles, performance_data, output_dir):
    """Compare all algorithms and generate summary plots and tables."""
    print("\nGenerating comparison plots and tables...")

    # Plot horizontal profiles
    with tqdm(total=100, desc="Plotting horizontal profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        plot_parabolic_profiles(
            horizontal_profiles,
            "Horizontal Parabolic Profiles Comparison",
            os.path.join(output_dir, "horizontal_profiles_comparison.png")
        )
        pbar.update(100)

    # Plot vertical profiles
    with tqdm(total=100, desc="Plotting vertical profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        plot_parabolic_profiles(
            vertical_profiles,
            "Vertical Parabolic Profiles Comparison",
            os.path.join(output_dir, "vertical_profiles_comparison.png")
        )
        pbar.update(100)

    # Create performance comparison table
    with tqdm(total=100, desc="Creating performance comparison", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        plt.figure(figsize=(10, 6))
        algorithms = list(performance_data.keys())
        execution_times = [performance_data[algo] for algo in algorithms]

        plt.bar(algorithms, execution_times)
        plt.title("Algorithm Performance Comparison")
        plt.xlabel("Algorithm")
        plt.ylabel("Execution Time (seconds)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=200)
        plt.close()
        pbar.update(50)

        # Save performance data to CSV
        with open(os.path.join(output_dir, "performance_comparison.csv"), 'w') as f:
            f.write("Algorithm,Execution Time (seconds)\n")
            for algo, time in performance_data.items():
                f.write(f"{algo},{time:.2f}\n")
        pbar.update(50)

def main():
    # Create output directory
    output_dir = 'benchmark_results_bits08'
    os.makedirs(output_dir, exist_ok=True)

    logging.info("\n=== Starting Benchmark of Non-OpenCL Algorithms on Bits08 Test Case ===")
    logging.info("This benchmark will run multiple optical flow algorithms and compare their results.")
    logging.info("Progress bars will show the status of each step.\n")

    # Load Bits08 test images
    logging.info("Loading test images...")
    with tqdm(total=100, desc="Preparing benchmark", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        img1, img2 = load_bits08_images()
        pbar.update(100)
    logging.info("Test images loaded successfully.")

    # Run all algorithms
    results = {}
    performance_data = {}
    horizontal_profiles = {}
    vertical_profiles = {}

    # Farneback
    logging.info("\n[1/6] Running Farneback_Numba algorithm...")
    U_fb, V_fb, time_fb = run_farneback(img1, img2, output_dir)
    results["Farneback_Numba"] = (U_fb, V_fb)
    performance_data["Farneback_Numba"] = time_fb
    logging.info(f"Farneback_Numba completed in {time_fb:.2f} seconds")

    # Extract profiles
    logging.info("Extracting Farneback profiles...")
    with tqdm(total=100, desc="Extracting Farneback profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        x_h, v_h = extract_parabolic_profile(U_fb, V_fb, axis='horizontal')
        pbar.update(50)
        x_v, v_v = extract_parabolic_profile(U_fb, V_fb, axis='vertical')
        pbar.update(50)
        horizontal_profiles["Farneback_Numba"] = (x_h, v_h)
        vertical_profiles["Farneback_Numba"] = (x_v, v_v)
    logging.info("Farneback profiles extracted successfully")

    # Farneback with Liu-Shen
    logging.info("\n[2/6] Running Farneback_Numba with Liu-Shen enhancement...")
    U_fb_ls, V_fb_ls, time_fb_ls = run_farneback_liushen(img1, img2, output_dir)
    results["Farneback_Numba_LiuShen"] = (U_fb_ls, V_fb_ls)
    performance_data["Farneback_Numba_LiuShen"] = time_fb_ls
    logging.info(f"Farneback_Numba_LiuShen completed in {time_fb_ls:.2f} seconds")

    # Extract profiles
    logging.info("Extracting Farneback+LiuShen profiles...")
    with tqdm(total=100, desc="Extracting Farneback+LiuShen profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        x_h, v_h = extract_parabolic_profile(U_fb_ls, V_fb_ls, axis='horizontal')
        pbar.update(50)
        x_v, v_v = extract_parabolic_profile(U_fb_ls, V_fb_ls, axis='vertical')
        pbar.update(50)
        horizontal_profiles["Farneback_Numba_LiuShen"] = (x_h, v_h)
        vertical_profiles["Farneback_Numba_LiuShen"] = (x_v, v_v)
    logging.info("Farneback+LiuShen profiles extracted successfully")

    # Dense Lucas-Kanade
    logging.info("\n[3/6] Running DenseLucasKanade_Numba algorithm...")
    U_lk, V_lk, time_lk = run_dense_lucas_kanade(img1, img2, output_dir)
    results["DenseLucasKanade_Numba"] = (U_lk, V_lk)
    performance_data["DenseLucasKanade_Numba"] = time_lk
    logging.info(f"DenseLucasKanade_Numba completed in {time_lk:.2f} seconds")

    # Extract profiles
    logging.info("Extracting Lucas-Kanade profiles...")
    with tqdm(total=100, desc="Extracting Lucas-Kanade profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        x_h, v_h = extract_parabolic_profile(U_lk, V_lk, axis='horizontal')
        pbar.update(50)
        x_v, v_v = extract_parabolic_profile(U_lk, V_lk, axis='vertical')
        pbar.update(50)
        horizontal_profiles["DenseLucasKanade_Numba"] = (x_h, v_h)
        vertical_profiles["DenseLucasKanade_Numba"] = (x_v, v_v)
    logging.info("Lucas-Kanade profiles extracted successfully")

    # Dense Lucas-Kanade with Liu-Shen
    logging.info("\n[4/6] Running DenseLucasKanade_Numba with Liu-Shen enhancement...")
    U_lk_ls, V_lk_ls, time_lk_ls = run_dense_lucas_kanade_liushen(img1, img2, output_dir)
    results["DenseLucasKanade_Numba_LiuShen"] = (U_lk_ls, V_lk_ls)
    performance_data["DenseLucasKanade_Numba_LiuShen"] = time_lk_ls
    logging.info(f"DenseLucasKanade_Numba_LiuShen completed in {time_lk_ls:.2f} seconds")

    # Extract profiles
    logging.info("Extracting Lucas-Kanade+LiuShen profiles...")
    with tqdm(total=100, desc="Extracting Lucas-Kanade+LiuShen profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        x_h, v_h = extract_parabolic_profile(U_lk_ls, V_lk_ls, axis='horizontal')
        pbar.update(50)
        x_v, v_v = extract_parabolic_profile(U_lk_ls, V_lk_ls, axis='vertical')
        pbar.update(50)
        horizontal_profiles["DenseLucasKanade_Numba_LiuShen"] = (x_h, v_h)
        vertical_profiles["DenseLucasKanade_Numba_LiuShen"] = (x_v, v_v)
    logging.info("Lucas-Kanade+LiuShen profiles extracted successfully")

    # Horn-Schunck
    logging.info("\n[5/6] Running Horn-Schunck algorithm...")
    U_hs, V_hs, time_hs = run_horn_schunck(img1, img2, output_dir)
    results["HornSchunck"] = (U_hs, V_hs)
    performance_data["HornSchunck"] = time_hs
    logging.info(f"Horn-Schunck completed in {time_hs:.2f} seconds")

    # Extract profiles
    logging.info("Extracting Horn-Schunck profiles...")
    with tqdm(total=100, desc="Extracting Horn-Schunck profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        x_h, v_h = extract_parabolic_profile(U_hs, V_hs, axis='horizontal')
        pbar.update(50)
        x_v, v_v = extract_parabolic_profile(U_hs, V_hs, axis='vertical')
        pbar.update(50)
        horizontal_profiles["HornSchunck"] = (x_h, v_h)
        vertical_profiles["HornSchunck"] = (x_v, v_v)
    logging.info("Horn-Schunck profiles extracted successfully")

    # Horn-Schunck with Liu-Shen
    logging.info("\n[6/6] Running Horn-Schunck with Liu-Shen enhancement...")
    U_hs_ls, V_hs_ls, time_hs_ls = run_horn_schunck_liushen(img1, img2, output_dir)
    results["HornSchunck_LiuShen"] = (U_hs_ls, V_hs_ls)
    performance_data["HornSchunck_LiuShen"] = time_hs_ls
    logging.info(f"Horn-Schunck with Liu-Shen completed in {time_hs_ls:.2f} seconds")

    # Extract profiles
    logging.info("Extracting Horn-Schunck+LiuShen profiles...")
    with tqdm(total=100, desc="Extracting Horn-Schunck+LiuShen profiles", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        x_h, v_h = extract_parabolic_profile(U_hs_ls, V_hs_ls, axis='horizontal')
        pbar.update(50)
        x_v, v_v = extract_parabolic_profile(U_hs_ls, V_hs_ls, axis='vertical')
        pbar.update(50)
        horizontal_profiles["HornSchunck_LiuShen"] = (x_h, v_h)
        vertical_profiles["HornSchunck_LiuShen"] = (x_v, v_v)
    logging.info("Horn-Schunck+LiuShen profiles extracted successfully")

    # Compare all algorithms
    logging.info("\nGenerating comparison plots and tables...")
    compare_all_algorithms(horizontal_profiles, vertical_profiles, performance_data, output_dir)

    logging.info(f"\nBenchmark completed. Results saved to {output_dir} directory.")
    logging.info("You can view the results in the following files:")
    logging.info(f"  - Flow field visualizations: {output_dir}/*.png")
    logging.info(f"  - Flow data: {output_dir}/*.mat")
    logging.info(f"  - Performance comparison: {output_dir}/performance_comparison.csv")

if __name__ == "__main__":
    main()
