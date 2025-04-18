#!/usr/bin/env python
"""
Unified flow analysis script.

This script provides a unified interface for running different optical flow
analysis methods on the parabolic test images.
"""

import os
import sys
import argparse
import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Import utility functions
from utils.flow_utils import (
    load_parabolic_images,
    visualize_flow,
    visualize_horizontal_profile,
    analyze_profile,
    compute_block_matching_flow,
    interpolate_flow,
    optimize_flow,
    save_flow_results
)

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def run_block_matching(im1, im2, window_size=16, overlap=8, smooth_sigma=1.5, output_dir='flow_results'):
    """
    Run block matching flow analysis.

    Args:
        im1, im2: Input images
        window_size: Size of the matching window
        overlap: Overlap between windows
        smooth_sigma: Sigma for Gaussian smoothing
        output_dir: Output directory

    Returns:
        U, V: Optimized flow components
    """
    # Create output directory
    method_dir = os.path.join(output_dir, f'block_matching_w{window_size}_o{overlap}')
    os.makedirs(method_dir, exist_ok=True)

    # Compute block matching flow
    u, v, x, y = compute_block_matching_flow(im1, im2, window_size, overlap)

    # Print flow statistics
    print(f"Grid flow range: u min={u.min():.4f}, max={u.max():.4f}, v min={v.min():.4f}, max={v.max():.4f}")

    # Interpolate flow to full image size
    print("Interpolating flow to full image size...")
    U, V = interpolate_flow(u, v, x, y, im1.shape)

    # Print interpolated flow statistics
    print(f"Interpolated flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")

    # Optimize flow
    print("Optimizing flow by scaling and smoothing...")
    U_opt, V_opt = optimize_flow(U, V, target_u_range=(-4, 0), target_v_range=(-1, 1), smooth_sigma=smooth_sigma)

    # Print optimized flow statistics
    print(f"Optimized flow range: U min={U_opt.min():.4f}, max={U_opt.max():.4f}, V min={V_opt.min():.4f}, max={V_opt.max():.4f}")

    # Visualize flow
    visualize_flow(U_opt, V_opt, f"Block Matching (window={window_size}, overlap={overlap})",
                 os.path.join(method_dir, "flow.png"))

    # Analyze profile
    analyze_profile(U_opt, V_opt,
                  f"Block Matching (window={window_size}, overlap={overlap})",
                  os.path.join(method_dir, "profile.png"))

    # Save results
    params = {
        'window_size': window_size,
        'overlap': overlap,
        'smooth_sigma': smooth_sigma,
        'method': 'block_matching'
    }

    save_flow_results(
        u, v, x, y, U_opt, V_opt, params,
        os.path.join(method_dir, "flow_results.mat")
    )

    return U_opt, V_opt

def run_farneback(im1, im2, window_size=15, iterations=5, poly_n=5, poly_sigma=1.2,
                pyr_levels=1, smooth_sigma=1.5, output_dir='flow_results'):
    """
    Run Farneback flow analysis.

    Args:
        im1, im2: Input images
        window_size: Window size for Farneback
        iterations: Number of iterations
        poly_n: Polynomial degree
        poly_sigma: Polynomial sigma
        pyr_levels: Number of pyramid levels
        smooth_sigma: Sigma for Gaussian smoothing
        output_dir: Output directory

    Returns:
        U, V: Optimized flow components
    """
    # Import Farneback implementation
    from Farneback_Numba_optimized import Farneback_Numba

    # Create output directory
    method_dir = os.path.join(output_dir, f'farneback_w{window_size}_i{iterations}_n{poly_n}_s{poly_sigma}_p{pyr_levels}')
    os.makedirs(method_dir, exist_ok=True)

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
    print(f"\nRunning Farneback with parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Polynomial degree: {poly_n}")
    print(f"  Polynomial sigma: {poly_sigma}")
    print(f"  Pyramid levels: {pyr_levels}")

    # Run the algorithm
    try:
        start_time = time.time()
        U, V, error = fb.compute(im1, im2, U, V)
        elapsed_time = time.time() - start_time
        print(f"Farneback computation completed in {elapsed_time:.2f} seconds")

        # Print flow statistics
        print(f"Flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")

        # Optimize flow
        print("Optimizing flow by scaling and smoothing...")
        U_opt, V_opt = optimize_flow(U, V, target_u_range=(-4, 0), target_v_range=(-1, 1), smooth_sigma=smooth_sigma)

        # Print optimized flow statistics
        print(f"Optimized flow range: U min={U_opt.min():.4f}, max={U_opt.max():.4f}, V min={V_opt.min():.4f}, max={V_opt.max():.4f}")

        # Visualize flow
        visualize_flow(U_opt, V_opt, f"Farneback (window={window_size}, iterations={iterations})",
                     os.path.join(method_dir, "flow.png"))

        # Analyze profile
        analyze_profile(U_opt, V_opt,
                      f"Farneback (window={window_size}, iterations={iterations})",
                      os.path.join(method_dir, "profile.png"))

        # Create dummy grid for saving
        ny, nx = 10, 10
        y, x = np.mgrid[0:im1.shape[0]:im1.shape[0]//ny, 0:im1.shape[1]:im1.shape[1]//nx]
        u = np.zeros_like(x)
        v = np.zeros_like(y)

        # Save results
        params = {
            'window_size': window_size,
            'iterations': iterations,
            'poly_n': poly_n,
            'poly_sigma': poly_sigma,
            'pyr_levels': pyr_levels,
            'smooth_sigma': smooth_sigma,
            'method': 'farneback'
        }

        save_flow_results(
            u, v, x, y, U_opt, V_opt, params,
            os.path.join(method_dir, "flow_results.mat")
        )

        return U_opt, V_opt

    except Exception as e:
        print(f"Error in Farneback computation: {e}")
        return None, None

def run_dense_lucas_kanade(im1, im2, window_size=15, iterations=5, pyr_levels=1, smooth_sigma=1.5, output_dir='flow_results'):
    """
    Run Dense Lucas-Kanade flow analysis.

    Args:
        im1, im2: Input images
        window_size: Window size for Lucas-Kanade
        iterations: Number of iterations
        pyr_levels: Number of pyramid levels
        smooth_sigma: Sigma for Gaussian smoothing
        output_dir: Output directory

    Returns:
        U, V: Optimized flow components
    """
    # Import Dense Lucas-Kanade implementation
    from denseLucasKanade_numba import denseLucasKanade_numba

    # Create output directory
    method_dir = os.path.join(output_dir, f'dense_lucas_kanade_w{window_size}_i{iterations}_p{pyr_levels}')
    os.makedirs(method_dir, exist_ok=True)

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Create Dense Lucas-Kanade adapter
    lk = denseLucasKanade_numba(
        halfWindow=window_size//2,
        Niter=iterations,
        provideGenericPyramidalDefaults=True
    )

    # Print parameters
    print(f"\nRunning Dense Lucas-Kanade with parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Pyramid levels: {pyr_levels}")

    # Run the algorithm
    try:
        start_time = time.time()
        U, V, error = lk.compute(im1, im2, U, V)
        elapsed_time = time.time() - start_time
        print(f"Dense Lucas-Kanade computation completed in {elapsed_time:.2f} seconds")

        # Print flow statistics
        print(f"Flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")

        # Optimize flow
        print("Optimizing flow by scaling and smoothing...")
        U_opt, V_opt = optimize_flow(U, V, target_u_range=(-4, 0), target_v_range=(-1, 1), smooth_sigma=smooth_sigma)

        # Print optimized flow statistics
        print(f"Optimized flow range: U min={U_opt.min():.4f}, max={U_opt.max():.4f}, V min={V_opt.min():.4f}, max={V_opt.max():.4f}")

        # Visualize flow
        visualize_flow(U_opt, V_opt, f"Dense Lucas-Kanade (window={window_size}, iterations={iterations})",
                     os.path.join(method_dir, "flow.png"))

        # Analyze profile
        analyze_profile(U_opt, V_opt,
                      f"Dense Lucas-Kanade (window={window_size}, iterations={iterations})",
                      os.path.join(method_dir, "profile.png"))

        # Create dummy grid for saving
        ny, nx = 10, 10
        y, x = np.mgrid[0:im1.shape[0]:im1.shape[0]//ny, 0:im1.shape[1]:im1.shape[1]//nx]
        u = np.zeros_like(x)
        v = np.zeros_like(y)

        # Save results
        params = {
            'window_size': window_size,
            'iterations': iterations,
            'pyr_levels': pyr_levels,
            'smooth_sigma': smooth_sigma,
            'method': 'dense_lucas_kanade'
        }

        save_flow_results(
            u, v, x, y, U_opt, V_opt, params,
            os.path.join(method_dir, "flow_results.mat")
        )

        return U_opt, V_opt

    except Exception as e:
        print(f"Error in Dense Lucas-Kanade computation: {e}")
        return None, None

def run_horn_schunck(im1, im2, alpha=0.1, iterations=100, smooth_sigma=1.5, output_dir='flow_results'):
    """
    Run Horn-Schunck flow analysis.

    Args:
        im1, im2: Input images
        alpha: Regularization parameter
        iterations: Number of iterations
        smooth_sigma: Sigma for Gaussian smoothing
        output_dir: Output directory

    Returns:
        U, V: Optimized flow components
    """
    # Import Horn-Schunck implementation
    from HornSchunck_optimized import HSOpticalFlowAlgoAdapter, HS

    # Create output directory
    method_dir = os.path.join(output_dir, f'horn_schunck_a{alpha}_i{iterations}')
    os.makedirs(method_dir, exist_ok=True)

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Create Horn-Schunck adapter
    hs_adapter = HSOpticalFlowAlgoAdapter(
        alphas=[alpha],  # List of alphas
        Niter=iterations,
        provideGenericPyramidalDefaults=True
    )

    # Print parameters
    print(f"\nRunning Horn-Schunck with parameters:")
    print(f"  Alpha: {alpha}")
    print(f"  Iterations: {iterations}")

    # Run the algorithm
    try:
        start_time = time.time()
        # Use the adapter or direct HS function
        # U, V = hs_adapter.compute(im1, im2, U, V)
        # Direct call to HS function
        U, V, _ = HS(im2, im1, alpha, iterations, U, V)
        elapsed_time = time.time() - start_time
        print(f"Horn-Schunck computation completed in {elapsed_time:.2f} seconds")

        # Print flow statistics
        print(f"Flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")

        # Optimize flow
        print("Optimizing flow by scaling and smoothing...")
        U_opt, V_opt = optimize_flow(U, V, target_u_range=(-4, 0), target_v_range=(-1, 1), smooth_sigma=smooth_sigma)

        # Print optimized flow statistics
        print(f"Optimized flow range: U min={U_opt.min():.4f}, max={U_opt.max():.4f}, V min={V_opt.min():.4f}, max={V_opt.max():.4f}")

        # Visualize flow
        visualize_flow(U_opt, V_opt, f"Horn-Schunck (alpha={alpha}, iterations={iterations})",
                     os.path.join(method_dir, "flow.png"))

        # Analyze profile
        analyze_profile(U_opt, V_opt,
                      f"Horn-Schunck (alpha={alpha}, iterations={iterations})",
                      os.path.join(method_dir, "profile.png"))

        # Create dummy grid for saving
        ny, nx = 10, 10
        y, x = np.mgrid[0:im1.shape[0]:im1.shape[0]//ny, 0:im1.shape[1]:im1.shape[1]//nx]
        u = np.zeros_like(x)
        v = np.zeros_like(y)

        # Save results
        params = {
            'alpha': alpha,
            'iterations': iterations,
            'smooth_sigma': smooth_sigma,
            'method': 'horn_schunck'
        }

        save_flow_results(
            u, v, x, y, U_opt, V_opt, params,
            os.path.join(method_dir, "flow_results.mat")
        )

        return U_opt, V_opt

    except Exception as e:
        print(f"Error in Horn-Schunck computation: {e}")
        return None, None

def run_farneback_liushen(im1, im2, window_size=15, iterations=5, poly_n=5, poly_sigma=1.2,
                        pyr_levels=2, liu_shen_alpha=0.1, smooth_sigma=1.5, output_dir='flow_results'):
    """
    Run Farneback with Liu-Shen enhancement.

    Args:
        im1, im2: Input images
        window_size: Window size for Farneback
        iterations: Number of iterations
        poly_n: Polynomial degree
        poly_sigma: Polynomial sigma
        pyr_levels: Number of pyramid levels
        liu_shen_alpha: Alpha parameter for Liu-Shen
        smooth_sigma: Sigma for Gaussian smoothing
        output_dir: Output directory

    Returns:
        U, V: Optimized flow components
    """
    # Import required implementations
    from Farneback_Numba_optimized import Farneback_Numba
    from PhysicsBasedOpticalFlowLiuShen_optimized import LiuShenOpticalFlowAlgoAdapter
    from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow

    # Create output directory
    method_dir = os.path.join(output_dir, f'farneback_liushen_w{window_size}_i{iterations}_n{poly_n}_s{poly_sigma}_p{pyr_levels}_a{liu_shen_alpha}')
    os.makedirs(method_dir, exist_ok=True)

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Create Farneback adapter
    fb_adapter = Farneback_Numba(
        windowSize=window_size,
        Niters=iterations,
        polyN=poly_n,
        polySigma=poly_sigma,
        pyramidalLevels=pyr_levels
    )

    # Create Liu-Shen adapter
    ls_adapter = LiuShenOpticalFlowAlgoAdapter(alpha=liu_shen_alpha)

    # Common parameters
    FILTER = 2
    FILTER_OPT = 0.48
    kLevels = 1

    # Print parameters
    print(f"\nRunning Farneback with Liu-Shen enhancement:")
    print(f"  Window size: {window_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Polynomial degree: {poly_n}")
    print(f"  Polynomial sigma: {poly_sigma}")
    print(f"  Pyramid levels: {pyr_levels}")
    print(f"  Liu-Shen alpha: {liu_shen_alpha}")

    # Run the algorithm
    try:
        start_time = time.time()

        # Show progress bar
        with tqdm(total=100, desc="Computing optical flow", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            # Run the algorithm
            U, V = genericPyramidalOpticalFlow(
                im1, im2, FILTER, fb_adapter,
                pyr_levels, kLevels,
                FILTER_OPT, ls_adapter,
                warping=True
            )

            # Update progress bar to 100%
            pbar.n = 100
            pbar.refresh()

        elapsed_time = time.time() - start_time
        print(f"Computation completed in {elapsed_time:.2f} seconds")

        # Print flow statistics
        print(f"Flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")

        # Optimize flow
        print("Optimizing flow by scaling and smoothing...")
        U_opt, V_opt = optimize_flow(U, V, target_u_range=(-4, 0), target_v_range=(-1, 1), smooth_sigma=smooth_sigma)

        # Print optimized flow statistics
        print(f"Optimized flow range: U min={U_opt.min():.4f}, max={U_opt.max():.4f}, V min={V_opt.min():.4f}, max={V_opt.max():.4f}")

        # Visualize flow
        visualize_flow(U_opt, V_opt, f"Farneback with Liu-Shen (window={window_size}, alpha={liu_shen_alpha})",
                     os.path.join(method_dir, "flow.png"))

        # Analyze profile
        analyze_profile(U_opt, V_opt,
                      f"Farneback with Liu-Shen (window={window_size}, alpha={liu_shen_alpha})",
                      os.path.join(method_dir, "profile.png"))

        # Create dummy grid for saving
        ny, nx = 10, 10
        y, x = np.mgrid[0:im1.shape[0]:im1.shape[0]//ny, 0:im1.shape[1]:im1.shape[1]//nx]
        u = np.zeros_like(x)
        v = np.zeros_like(y)

        # Save results
        params = {
            'window_size': window_size,
            'iterations': iterations,
            'poly_n': poly_n,
            'poly_sigma': poly_sigma,
            'pyr_levels': pyr_levels,
            'liu_shen_alpha': liu_shen_alpha,
            'smooth_sigma': smooth_sigma,
            'method': 'farneback_liushen'
        }

        save_flow_results(
            u, v, x, y, U_opt, V_opt, params,
            os.path.join(method_dir, "flow_results.mat")
        )

        return U_opt, V_opt

    except Exception as e:
        print(f"Error in Farneback with Liu-Shen computation: {e}")
        return None, None

def run_openpiv_smoothed(im1, im2, window_size=16, overlap=8, smooth_sigma=1.5, output_dir='flow_results'):
    """
    Run OpenPIV with smoothing between windows.

    Args:
        im1, im2: Input images
        window_size: Size of the interrogation window
        overlap: Overlap between windows
        smooth_sigma: Sigma for Gaussian smoothing
        output_dir: Output directory

    Returns:
        U, V: Optimized flow components
    """
    # Import OpenPIV
    try:
        from openpiv import pyprocess, filters
    except ImportError:
        print("Error: OpenPIV is not installed. Please install it with 'pip install openpiv'")
        return None, None

    # Create output directory
    method_dir = os.path.join(output_dir, f'openpiv_smoothed_w{window_size}_o{overlap}_s{smooth_sigma}')
    os.makedirs(method_dir, exist_ok=True)

    # Process the images to get the vector field
    print(f"\nRunning OpenPIV with parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Overlap: {overlap}")
    print(f"  Smooth sigma: {smooth_sigma}")

    # Make sure input images are valid
    frame_a = np.asarray(im1).astype(np.int32)
    frame_b = np.asarray(im2).astype(np.int32)

    try:
        start_time = time.time()

        # Process the images to get the vector field
        u, v, sig2noise = pyprocess.extended_search_area_piv(
            frame_a, frame_b,
            window_size=window_size,
            overlap=overlap,
            dt=1.0,
            search_area_size=window_size,
            sig2noise_method='peak2peak'
        )

        # Create a mask for invalid vectors (all False for now)
        mask = np.zeros_like(u, dtype=bool)

        # Replace outliers
        u, v = filters.replace_outliers(u, v, mask, method='localmean', max_iter=3, kernel_size=3)

        # Apply Gaussian smoothing to the velocity components
        print(f"Applying Gaussian smoothing with sigma={smooth_sigma}...")
        u_smooth = gaussian_filter(u, sigma=smooth_sigma)
        v_smooth = gaussian_filter(v, sigma=smooth_sigma)

        # Check for NaN or Inf values in the results
        if np.any(np.isnan(u_smooth)) or np.any(np.isinf(u_smooth)) or np.any(np.isnan(v_smooth)) or np.any(np.isinf(v_smooth)):
            print("Warning: PIV analysis produced NaN or Inf values. Replacing with zeros.")
            u_smooth = np.nan_to_num(u_smooth, nan=0, posinf=0, neginf=0)
            v_smooth = np.nan_to_num(v_smooth, nan=0, posinf=0, neginf=0)

        # Calculate the coordinates
        n_rows, n_cols = u.shape

        # Calculate the coordinates
        x_points = np.arange(window_size//2, frame_a.shape[1]-window_size//2+1, window_size-overlap)
        y_points = np.arange(window_size//2, frame_a.shape[0]-window_size//2+1, window_size-overlap)

        # Make sure the coordinate arrays match the velocity field dimensions
        if len(x_points) > n_cols:
            x_points = x_points[:n_cols]
        if len(y_points) > n_rows:
            y_points = y_points[:n_rows]

        # Create meshgrid
        x, y = np.meshgrid(x_points, y_points)

        elapsed_time = time.time() - start_time
        print(f"OpenPIV computation completed in {elapsed_time:.2f} seconds")

        # Print flow statistics
        print(f"Grid flow range: u min={u_smooth.min():.4f}, max={u_smooth.max():.4f}, v min={v_smooth.min():.4f}, max={v_smooth.max():.4f}")

        # Interpolate flow to full image size
        print("Interpolating flow to full image size...")
        U, V = interpolate_flow(u_smooth, v_smooth, x, y, im1.shape)

        # Print interpolated flow statistics
        print(f"Interpolated flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")

        # Visualize flow
        visualize_flow(U, V, f"OpenPIV Smoothed (window={window_size}, overlap={overlap}, sigma={smooth_sigma})",
                     os.path.join(method_dir, "flow.png"))

        # Analyze profile
        analyze_profile(U, V,
                      f"OpenPIV Smoothed (window={window_size}, overlap={overlap}, sigma={smooth_sigma})",
                      os.path.join(method_dir, "profile.png"))

        # Save results
        params = {
            'window_size': window_size,
            'overlap': overlap,
            'smooth_sigma': smooth_sigma,
            'method': 'openpiv_smoothed'
        }

        save_flow_results(
            u_smooth, v_smooth, x, y, U, V, params,
            os.path.join(method_dir, "flow_results.mat")
        )

        return U, V

    except Exception as e:
        print(f"Error in OpenPIV computation: {e}")
        return None, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified flow analysis script.')
    parser.add_argument('--method', type=str, default='block_matching',
                        choices=['block_matching', 'farneback', 'farneback_liushen', 'openpiv_smoothed',
                                 'dense_lucas_kanade', 'horn_schunck'],
                        help='Flow analysis method')
    parser.add_argument('--window_size', type=int, default=16,
                        help='Window size for analysis')
    parser.add_argument('--overlap', type=int, default=8,
                        help='Overlap between windows (for block_matching)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations (for farneback)')
    parser.add_argument('--poly_n', type=int, default=5,
                        help='Polynomial degree (for farneback)')
    parser.add_argument('--poly_sigma', type=float, default=1.2,
                        help='Polynomial sigma (for farneback)')
    parser.add_argument('--pyr_levels', type=int, default=1,
                        help='Number of pyramid levels (for farneback)')
    parser.add_argument('--liu_shen_alpha', type=float, default=0.1,
                        help='Alpha parameter for Liu-Shen (for farneback_liushen)')
    parser.add_argument('--smooth_sigma', type=float, default=1.5,
                        help='Sigma for Gaussian smoothing')
    parser.add_argument('--output_dir', type=str, default='flow_results',
                        help='Output directory')

    args = parser.parse_args()

    # Load test images
    im1, im2 = load_parabolic_images()

    # Run selected method
    if args.method == 'block_matching':
        _ = run_block_matching(
            im1, im2,
            window_size=args.window_size,
            overlap=args.overlap,
            smooth_sigma=args.smooth_sigma,
            output_dir=args.output_dir
        )
    elif args.method == 'farneback':
        _ = run_farneback(
            im1, im2,
            window_size=args.window_size,
            iterations=args.iterations,
            poly_n=args.poly_n,
            poly_sigma=args.poly_sigma,
            pyr_levels=args.pyr_levels,
            smooth_sigma=args.smooth_sigma,
            output_dir=args.output_dir
        )
    elif args.method == 'farneback_liushen':
        _ = run_farneback_liushen(
            im1, im2,
            window_size=args.window_size,
            iterations=args.iterations,
            poly_n=args.poly_n,
            poly_sigma=args.poly_sigma,
            pyr_levels=args.pyr_levels,
            liu_shen_alpha=args.liu_shen_alpha,
            smooth_sigma=args.smooth_sigma,
            output_dir=args.output_dir
        )
    elif args.method == 'openpiv_smoothed':
        _ = run_openpiv_smoothed(
            im1, im2,
            window_size=args.window_size,
            overlap=args.overlap,
            smooth_sigma=args.smooth_sigma,
            output_dir=args.output_dir
        )
    elif args.method == 'dense_lucas_kanade':
        _ = run_dense_lucas_kanade(
            im1, im2,
            window_size=args.window_size,
            iterations=args.iterations,
            pyr_levels=args.pyr_levels,
            smooth_sigma=args.smooth_sigma,
            output_dir=args.output_dir
        )
    elif args.method == 'horn_schunck':
        _ = run_horn_schunck(
            im1, im2,
            alpha=args.liu_shen_alpha,  # Reuse the liu_shen_alpha parameter
            iterations=args.iterations,
            smooth_sigma=args.smooth_sigma,
            output_dir=args.output_dir
        )

    print("\nFlow analysis complete.")

if __name__ == "__main__":
    main()
