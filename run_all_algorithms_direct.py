#!/usr/bin/env python
"""
Run all optical flow algorithms one by one and show results as they come in.
This version directly calls the algorithm functions from the src/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.io import savemat, loadmat
from tqdm import tqdm
from skimage.io import imread

# Import algorithm implementations
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow
from denseLucasKanade_Numba_fixed import denseLucasKanade_Numba
from Farneback_Numba import Farneback_Numba
from HornSchunck import HSOpticalFlowAlgoAdapter
from PhysicsBasedOpticalFlowLiuShen import LiuShenOpticalFlowAlgoAdapter

def load_bits08_images():
    """Load the Bits08 test images."""
    basePath = os.path.join('testImages', 'Bits08', 'Ni06')
    fn1 = os.path.join(basePath, 'parabolic01_0.tif')
    fn2 = os.path.join(basePath, 'parabolic01_1.tif')

    print(f"Loading images from {fn1} and {fn2}")

    # Load images
    im1 = imread(fn1).astype(np.float32)
    im2 = imread(fn2).astype(np.float32)

    # Normalize if needed
    if np.max(im1) > 1.0:
        im1 = im1 / np.max(im1)
    if np.max(im2) > 1.0:
        im2 = im2 / np.max(im2)

    return im1, im2

def save_flow(U, V, filename):
    """Save flow field to a .mat file."""
    margins = {'top': 0, 'left': 0, 'bottom': 0, 'right': 0}
    results = {'u': U, 'v': V, 'iaWidth': 1, 'iaHeight': 1, 'margins': margins}
    parameters = {'overlapFactor': 1.0, 'imageHeight': np.size(U, 0), 'imageWidth': np.size(U, 1)}

    savemat(filename, mdict={'velocities': results, 'parameters': parameters})
    print(f"Flow saved to {filename}")

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

def plot_flow_field(U, V, title, output_filename):
    """Plot optical flow as colormesh and quiver plots."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Auto-calculate color limits
    v_abs_max = max(abs(np.percentile(V, 1)), abs(np.percentile(V, 99)))
    vmin = -v_abs_max
    vmax = v_abs_max

    # Plot vertical velocity component as colormesh with high contrast colormap
    im1 = ax1.imshow(V, cmap='jet', vmin=vmin, vmax=vmax)
    ax1.set_title(f'{title} - Vertical Velocity (v)')
    plt.colorbar(im1, ax=ax1, label='Pixels/frame')

    # Add a grid for better reference
    ax1.grid(False)

    # Create quiver plot with color-coded arrows based on magnitude
    quiver_skip = 20
    y, x = np.mgrid[0:U.shape[0]:quiver_skip, 0:U.shape[1]:quiver_skip]
    u_skip = U[::quiver_skip, ::quiver_skip]
    v_skip = V[::quiver_skip, ::quiver_skip]

    # Calculate magnitude for coloring
    magnitude = np.sqrt(u_skip**2 + v_skip**2)

    # Plot quiver with colors based on magnitude
    quiv = ax2.quiver(x, y, u_skip, v_skip, magnitude,
                     scale=50, scale_units='inches',
                     cmap='jet', clim=[0, np.percentile(magnitude, 95)])
    plt.colorbar(quiv, ax=ax2, label='Magnitude (pixels/frame)')

    ax2.set_title(f'{title} - Vector Field')
    ax2.set_xlim(0, U.shape[1])
    ax2.set_ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

def plot_parabolic_profiles(U, V, title, output_filename):
    """Plot parabolic profiles."""
    # Extract profiles
    x_h, v_h = extract_parabolic_profile(U, V, axis='horizontal')
    x_v, v_v = extract_parabolic_profile(U, V, axis='vertical')

    plt.figure(figsize=(12, 5))

    # Plot horizontal profile
    plt.subplot(1, 2, 1)
    plt.plot(x_h, v_h)
    plt.title('Horizontal Parabolic Profile')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)

    # Plot vertical profile
    plt.subplot(1, 2, 2)
    plt.plot(x_v, v_v)
    plt.title('Vertical Parabolic Profile')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

    return (x_h, v_h), (x_v, v_v)

def run_algorithm(algorithm_name, im1, im2, params):
    """Run an algorithm and process its results."""
    print(f"\n{'='*80}")
    print(f"Running {algorithm_name}...")
    print(f"{'='*80}")

    # Create output directory
    output_dir = 'algorithm_results'
    os.makedirs(output_dir, exist_ok=True)

    # Run the algorithm
    start_time = time.time()

    # Common parameters
    FILTER = 2
    FILTER_OPT = 0.48
    pyramidalLevels = 2
    kLevels = 1

    # Initialize flow fields
    U = np.zeros(im1.shape, dtype=np.float32)
    V = np.zeros(im1.shape, dtype=np.float32)

    # Create algorithm adapter based on algorithm name
    if algorithm_name.startswith('DenseLucasKanade'):
        mainAdapter = denseLucasKanade_Numba(
            Niter=5,
            halfWindow=13,
            provideGenericPyramidalDefaults=True
        )
    elif algorithm_name.startswith('Farneback'):
        mainAdapter = Farneback_Numba(
            windowSize=33,
            Niters=5,
            polyN=7,
            polySigma=1.5,
            pyramidalLevels=3
        )
    elif algorithm_name.startswith('HornSchunck'):
        mainAdapter = HSOpticalFlowAlgoAdapter(
            alphas=[0.1],
            Niter=100,
            provideGenericPyramidalDefaults=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Create Liu-Shen adapter if needed
    if '_LiuShen' in algorithm_name:
        lsAdapter = LiuShenOpticalFlowAlgoAdapter(0.1)
    else:
        lsAdapter = None

    # Show progress bar while running
    with tqdm(total=100, desc=f"Running {algorithm_name}", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        # Run the algorithm
        [U, V] = genericPyramidalOpticalFlow(
            im1, im2, FILTER, mainAdapter,
            pyramidalLevels, kLevels,
            FILTER_OPT, lsAdapter,
            warping=False
        )

        # Update progress bar to 100%
        pbar.n = 100
        pbar.refresh()

    elapsed_time = time.time() - start_time
    print(f"{algorithm_name} completed in {elapsed_time:.2f} seconds")

    # Save the results
    output_file = os.path.join(output_dir, f"{algorithm_name}.mat")
    save_flow(U, V, output_file)

    print(f"Flow field shape: {U.shape}")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")

    # Plot flow field
    print(f"Plotting flow field for {algorithm_name}...")
    plot_flow_field(U, V, algorithm_name, os.path.join(output_dir, f"{algorithm_name}_flow.png"))

    # Plot parabolic profiles
    print(f"Plotting parabolic profiles for {algorithm_name}...")
    horizontal_profile, vertical_profile = plot_parabolic_profiles(
        U, V, algorithm_name, os.path.join(output_dir, f"{algorithm_name}_profiles.png")
    )

    return {
        'U': U,
        'V': V,
        'horizontal_profile': horizontal_profile,
        'vertical_profile': vertical_profile,
        'execution_time': elapsed_time
    }

def compare_all_algorithms(results):
    """Compare all algorithms and generate summary plots."""
    print("\nGenerating comparison plots...")

    # Create output directory
    output_dir = 'algorithm_results'
    os.makedirs(output_dir, exist_ok=True)

    # Plot horizontal profiles comparison
    plt.figure(figsize=(12, 8))

    for algo_name, result in results.items():
        if result:
            x, v = result['horizontal_profile']
            plt.plot(x, v, label=algo_name)

    plt.title("Horizontal Parabolic Profiles Comparison")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Vertical Velocity (pixels/frame)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "horizontal_profiles_comparison.png"), dpi=200)
    plt.close()

    # Plot vertical profiles comparison
    plt.figure(figsize=(12, 8))

    for algo_name, result in results.items():
        if result:
            x, v = result['vertical_profile']
            plt.plot(x, v, label=algo_name)

    plt.title("Vertical Parabolic Profiles Comparison")
    plt.xlabel("Position (pixels)")
    plt.ylabel("Vertical Velocity (pixels/frame)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vertical_profiles_comparison.png"), dpi=200)
    plt.close()

    # Create performance comparison table
    plt.figure(figsize=(10, 6))
    algorithms = list(results.keys())
    execution_times = [results[algo]['execution_time'] if results[algo] else 0 for algo in algorithms]

    plt.bar(algorithms, execution_times)
    plt.title("Algorithm Performance Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Execution Time (seconds)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=200)
    plt.close()

    # Save performance data to CSV
    with open(os.path.join(output_dir, "performance_comparison.csv"), 'w') as f:
        f.write("Algorithm,Execution Time (seconds)\n")
        for algo in algorithms:
            time_value = results[algo]['execution_time'] if results[algo] else 0
            f.write(f"{algo},{time_value:.2f}\n")

def main():
    # Load test images
    im1, im2 = load_bits08_images()

    # Define algorithms to run
    algorithms = [
        "DenseLucasKanade",
        "DenseLucasKanade_LiuShen",
        "Farneback",
        "Farneback_LiuShen",
        "HornSchunck",
        "HornSchunck_LiuShen"
    ]

    # Run each algorithm and collect results
    results = {}

    for algorithm_name in algorithms:
        params = {}  # Additional parameters if needed
        result = run_algorithm(algorithm_name, im1, im2, params)
        results[algorithm_name] = result

    # Compare all algorithms
    compare_all_algorithms(results)

    print("\nAll algorithms completed. Results saved to algorithm_results directory.")

if __name__ == "__main__":
    main()
