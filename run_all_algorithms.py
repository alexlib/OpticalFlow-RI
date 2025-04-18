#!/usr/bin/env python
"""
Run all optical flow algorithms one by one and show results as they come in.
"""

import sys
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.io import loadmat
from tqdm import tqdm

# Check if PyOpenCL is installed and available
try:
    import pyopencl
    # Try to get platforms to check if OpenCL is actually available
    try:
        platforms = pyopencl.get_platforms()
        if len(platforms) > 0:
            has_opencl = True
            print("PyOpenCL is installed and OpenCL platforms are available.")
        else:
            has_opencl = False
            print("PyOpenCL is installed but no OpenCL platforms were found.")
    except Exception as e:
        has_opencl = False
        print(f"PyOpenCL error: {e}. Will use Numba implementations only.")
except ImportError:
    has_opencl = False
    print("PyOpenCL is not installed. Will use Numba implementations only.")

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

def run_algorithm(script_name, algorithm_name):
    """Run an algorithm and process its results."""
    print(f"\n{'='*80}")
    print(f"Running {algorithm_name}...")
    print(f"{'='*80}")

    # If OpenCL is not available, modify the script to use Numba
    if not has_opencl and ('_PyCL' in script_name or 'Farneback' in script_name or 'denseLK' in script_name):
        print(f"OpenCL is not available. Modifying {script_name} to use Numba implementation...")

        # Create a modified version of the script that forces Numba
        numba_script = script_name.replace('.py', '_numba.py')
        with open(script_name, 'r') as f:
            content = f.read()

        # Force use_opencl to False
        content = content.replace('use_opencl = True', 'use_opencl = False')

        with open(numba_script, 'w') as f:
            f.write(content)

        # Use the modified script
        script_name = numba_script

    # Run the algorithm
    start_time = time.time()

    # Show progress bar while running
    with tqdm(total=100, desc=f"Running {algorithm_name}", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        process = subprocess.Popen(['python', script_name],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)

        # Update progress bar while process is running
        while process.poll() is None:
            time.sleep(0.5)
            # We don't know the exact progress, so we'll just update incrementally
            pbar.update(1)
            if pbar.n >= 99:
                pbar.n = 90  # Reset to avoid going over 100

        # Ensure we reach 100% when done
        pbar.n = 100
        pbar.refresh()

    # Get the output
    stdout, stderr = process.communicate()

    # Print any output
    if stdout:
        print("Output:")
        print(stdout)

    if stderr:
        print("Errors:")
        print(stderr)

    elapsed_time = time.time() - start_time
    print(f"{algorithm_name} completed in {elapsed_time:.2f} seconds")

    # Load the results
    mat_file = f"{script_name.replace('.py', '.mat')}"
    print(f"Loading results from {mat_file}")

    try:
        data = loadmat(mat_file)
        U = data['velocities']['u'][0, 0]
        V = data['velocities']['v'][0, 0]

        print(f"Flow field shape: {U.shape}")
        print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
        print(f"Flow range V: {V.min():.2f}                                                                                          to {V.max():.2f}")

        # Create output directory
        output_dir = 'algorithm_results'
        os.makedirs(output_dir, exist_ok=True)

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

    except Exception as e:
        print(f"Error processing results: {e}")
        return None

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
    # plt.legend()
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
    # Define algorithms to run
    algorithms = [
        # ("denseLK_Fs2_0_PyrLvls2.py", "DenseLucasKanade"),
        ("denseLK_Fs2_0_PyrLvls2_LiuShen.py", "DenseLucasKanade_LiuShen"),
        # ("Farneback_Fs2_0_PyrLvls2.py", "Farneback"),
        ("Farneback_Fs2_0_PyrLvls2_LiuShen.py", "Farneback_LiuShen"),
        # ("HornSchunck_Fs2_0_PyrLvls2.py", "HornSchunck"),
        ("HornSchunck_Fs2_0_PyrLvls2_LiuShen.py", "HornSchunck_LiuShen")
    ]

    # Run each algorithm and collect results
    results = {}

    for script_name, algorithm_name in algorithms:
        result = run_algorithm(script_name, algorithm_name)
        results[algorithm_name] = result

    # Compare all algorithms
    compare_all_algorithms(results)

    print("\nAll algorithms completed. Results saved to algorithm_results directory.")

if __name__ == "__main__":
    main()
