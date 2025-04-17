#!/usr/bin/env python
"""
Benchmark script for comparing optical flow methods in OpticalFlow-RI
using BOS (Background-Oriented Schlieren) images.

This script:
1. Loads a pair of BOS images
2. Runs all implemented optical flow algorithms with various parameters
3. Measures execution time
4. Generates visualizations (colormesh and quiver plots)
"""

import os
import sys
import time
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scipy.io

# Add the OpticalFlow-RI source directory to the path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import optical flow algorithms
from GenericPyramidalOpticalFlowWrapper import GenericPyramidalOpticalFlowWrapper
from PhysicsBasedOpticalFlowLiuShen import LiuShenOpticalFlowAlgoAdapter
from HornSchunck import HSOpticalFlowAlgoAdapter
from denseLucasKanade_PyCL import denseLucasKanade_PyCl
from Farneback_PyCL import Farneback_PyCL

# Function to save flow results
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

# Function to plot results
def plot_results(U, V, title, output_filename_base, vmin=None, vmax=None, quiver_scale=50, quiver_skip=40):
    """
    Plot optical flow results as colormesh and quiver plots

    Args:
        U, V: Horizontal and vertical velocity components
        title: Plot title
        output_filename_base: Base filename for saving plots
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
    plt.savefig(f'{output_filename_base}.png', dpi=200)
    plt.close()

# Main benchmark function
def run_benchmark(img1_path, img2_path, output_dir='benchmark_results'):
    """
    Run benchmark on all optical flow methods with various parameters

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

    # Results dictionary to store all benchmark results
    results = {}

    # 1. Horn-Schunck with different parameters
    hs_configs = [
        {'name': 'HS_Fs0_0', 'filter_sigma': 0.0, 'pyr_levels': 1, 'use_liu_shen': False},
        {'name': 'HS_Fs3_4', 'filter_sigma': 3.4, 'pyr_levels': 1, 'use_liu_shen': False},
        {'name': 'HS_Fs3_4_PyrLvls2', 'filter_sigma': 3.4, 'pyr_levels': 2, 'use_liu_shen': False},
        {'name': 'LiuSE_HS_Fs3_4_PyrLvls2', 'filter_sigma': 3.4, 'pyr_levels': 2, 'use_liu_shen': True}
    ]

    for config in hs_configs:
        print(f"\nRunning Horn-Schunck: {config['name']}")

        # Create Horn-Schunck adapter with enough alphas for all iterations
        # Each pyramidal level and k-level iteration will pop one alpha
        num_alphas = config['pyr_levels'] * 1  # 1 k-level
        hs_adapter = HSOpticalFlowAlgoAdapter(alphas=[1.0] * num_alphas, Niter=100, provideGenericPyramidalDefaults=True)

        # Create Liu-Shen adapter if needed
        if config['use_liu_shen']:
            # Liu-Shen adapter takes a single parameter 'alpha' which is the regularization parameter
            algo_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)
        else:
            algo_adapter = hs_adapter

        # Create pyramidal optical flow
        pyr_of = GenericPyramidalOpticalFlowWrapper(
            algo_adapter,
            filter_sigma=config['filter_sigma'],
            pyr_levels=config['pyr_levels']
        )

        # Run and time the algorithm
        start_time = time.time()
        U, V = pyr_of.calculateFlow(img1, img2)
        elapsed_time = time.time() - start_time

        # Store results
        results[config['name']] = {
            'U': U,
            'V': V,
            'time': elapsed_time,
            'config': config
        }

        print(f"  Execution time: {elapsed_time:.2f} seconds")
        print(f"  Flow range U: {U.min():.2f} to {U.max():.2f}")
        print(f"  Flow range V: {V.min():.2f} to {V.max():.2f}")

        # Save flow data
        save_flow(U, V, os.path.join(output_dir, f"{config['name']}.mat"))

        # Plot results
        plot_results(U, V, f"Horn-Schunck {config['name']}",
                    os.path.join(output_dir, f"{config['name']}"))

    # 2. Dense Lucas-Kanade with different parameters
    lk_configs = [
        {'name': 'LK_Fs2_0', 'filter_sigma': 2.0, 'pyr_levels': 1, 'use_liu_shen': False, 'half_window': 13},
        {'name': 'LK_Fs2_0_PyrLvls2', 'filter_sigma': 2.0, 'pyr_levels': 2, 'use_liu_shen': False, 'half_window': 13},
        {'name': 'LiuSE_LK_Fs2_0_PyrLvls2', 'filter_sigma': 2.0, 'pyr_levels': 2, 'use_liu_shen': True, 'half_window': 13}
    ]

    for config in lk_configs:
        print(f"\nRunning Dense Lucas-Kanade: {config['name']}")

        try:
            # Create Lucas-Kanade adapter
            lk_adapter = denseLucasKanade_PyCl(halfWindow=config['half_window'], Niter=5)

            # Create Liu-Shen adapter if needed
            if config['use_liu_shen']:
                # Liu-Shen adapter takes a single parameter 'alpha' which is the regularization parameter
                algo_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)
            else:
                algo_adapter = lk_adapter

            # Create pyramidal optical flow
            pyr_of = GenericPyramidalOpticalFlowWrapper(
                algo_adapter,
                filter_sigma=config['filter_sigma'],
                pyr_levels=config['pyr_levels']
            )

            # Run and time the algorithm
            start_time = time.time()
            U, V = pyr_of.calculateFlow(img1, img2)
            elapsed_time = time.time() - start_time

            # Store results
            results[config['name']] = {
                'U': U,
                'V': V,
                'time': elapsed_time,
                'config': config
            }

            print(f"  Execution time: {elapsed_time:.2f} seconds")
            print(f"  Flow range U: {U.min():.2f} to {U.max():.2f}")
            print(f"  Flow range V: {V.min():.2f} to {V.max():.2f}")

            # Save flow data
            save_flow(U, V, os.path.join(output_dir, f"{config['name']}.mat"))

            # Plot results
            plot_results(U, V, f"Lucas-Kanade {config['name']}",
                        os.path.join(output_dir, f"{config['name']}"))
        except Exception as e:
            print(f"  Error running Lucas-Kanade {config['name']}: {e}")

    # 3. Farneback with different parameters
    fb_configs = [
        {'name': 'FB_Fs0_0', 'filter_sigma': 0.0, 'pyr_levels': 1, 'use_liu_shen': False, 'window_size': 33},
        {'name': 'FB_Fs0_0_PyrLvls2', 'filter_sigma': 0.0, 'pyr_levels': 2, 'use_liu_shen': False, 'window_size': 33},
        {'name': 'LiuSE_FB_Fs0_0_PyrLvls2', 'filter_sigma': 0.0, 'pyr_levels': 2, 'use_liu_shen': True, 'window_size': 33}
    ]

    for config in fb_configs:
        print(f"\nRunning Farneback: {config['name']}")

        try:
            # Create Farneback adapter
            fb_adapter = Farneback_PyCL(windowSize=config['window_size'], Niters=5, polyN=7, polySigma=1.5)

            # Create Liu-Shen adapter if needed
            if config['use_liu_shen']:
                # Liu-Shen adapter takes a single parameter 'alpha' which is the regularization parameter
                algo_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)
            else:
                algo_adapter = fb_adapter

            # Create pyramidal optical flow
            pyr_of = GenericPyramidalOpticalFlowWrapper(
                algo_adapter,
                filter_sigma=config['filter_sigma'],
                pyr_levels=config['pyr_levels']
            )

            # Run and time the algorithm
            start_time = time.time()
            U, V = pyr_of.calculateFlow(img1, img2)
            elapsed_time = time.time() - start_time

            # Store results
            results[config['name']] = {
                'U': U,
                'V': V,
                'time': elapsed_time,
                'config': config
            }

            print(f"  Execution time: {elapsed_time:.2f} seconds")
            print(f"  Flow range U: {U.min():.2f} to {U.max():.2f}")
            print(f"  Flow range V: {V.min():.2f} to {V.max():.2f}")

            # Save flow data
            save_flow(U, V, os.path.join(output_dir, f"{config['name']}.mat"))

            # Plot results
            plot_results(U, V, f"Farneback {config['name']}",
                        os.path.join(output_dir, f"{config['name']}"))
        except Exception as e:
            print(f"  Error running Farneback {config['name']}: {e}")

    # Create summary plot of execution times
    method_names = list(results.keys())
    execution_times = [results[name]['time'] for name in method_names]

    plt.figure(figsize=(12, 6))
    plt.bar(method_names, execution_times)
    plt.xlabel('Method')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Optical Flow Methods - Execution Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time_comparison.png'), dpi=150)
    plt.close()

    # Create summary table
    with open(os.path.join(output_dir, 'benchmark_summary.txt'), 'w') as f:
        f.write("Optical Flow Methods Benchmark Summary\n")
        f.write("=====================================\n\n")
        f.write(f"{'Method':<30} {'Time (s)':<10} {'U min/max':<20} {'V min/max':<20}\n")
        f.write("-" * 80 + "\n")

        for name in method_names:
            result = results[name]
            u_range = f"{result['U'].min():.2f}/{result['U'].max():.2f}"
            v_range = f"{result['V'].min():.2f}/{result['V'].max():.2f}"
            f.write(f"{name:<30} {result['time']:<10.2f} {u_range:<20} {v_range:<20}\n")

    return results

if __name__ == "__main__":
    # Define image paths
    img1_path = "/home/user/Documents/repos/1pxPIV/Data/Test BOS Cropped/11-49-28.000-4.tif"
    img2_path = "/home/user/Documents/repos/1pxPIV/Data/Test BOS Cropped/11-49-28.000-6.tif"

    # Run benchmark
    results = run_benchmark(img1_path, img2_path)

    print("\nBenchmark completed. Results saved to 'benchmark_results' directory.")
