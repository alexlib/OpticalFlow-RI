#!/usr/bin/env python
"""
Run all available optical flow methods.

This script runs all the available optical flow methods using the unified
flow_analysis.py script with the specified parameters.
"""

import os
import sys
import subprocess
import argparse
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run all available optical flow methods.')
    parser.add_argument('--window_size', type=int, default=16,
                        help='Window size for analysis')
    parser.add_argument('--overlap', type=int, default=8,
                        help='Overlap between windows')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations (for Farneback)')
    parser.add_argument('--poly_n', type=int, default=5,
                        help='Polynomial degree (for Farneback)')
    parser.add_argument('--poly_sigma', type=float, default=1.2,
                        help='Polynomial sigma (for Farneback)')
    parser.add_argument('--pyr_levels', type=int, default=1,
                        help='Number of pyramid levels (for Farneback)')
    parser.add_argument('--liu_shen_alpha', type=float, default=0.1,
                        help='Alpha parameter for Liu-Shen')
    parser.add_argument('--smooth_sigma', type=float, default=1.5,
                        help='Sigma for Gaussian smoothing')
    parser.add_argument('--output_dir', type=str, default='flow_results',
                        help='Output directory')

    args = parser.parse_args()

    # List of available methods
    methods = [
        "block_matching",
        "farneback",
        "farneback_liushen",
        "openpiv_smoothed",
        "dense_lucas_kanade",
        "horn_schunck"
    ]

    # Run each method
    results = {}
    for method in tqdm(methods, desc="Running methods"):
        print(f"\nRunning {method} method...")

        # Build command
        cmd = [
            "python", "flow_analysis.py",
            "--method", method,
            "--window_size", str(args.window_size),
            "--overlap", str(args.overlap),
            "--iterations", str(args.iterations),
            "--poly_n", str(args.poly_n),
            "--poly_sigma", str(args.poly_sigma),
            "--pyr_levels", str(args.pyr_levels),
            "--liu_shen_alpha", str(args.liu_shen_alpha),
            "--smooth_sigma", str(args.smooth_sigma),
            "--output_dir", str(args.output_dir)
        ]

        # Run command
        try:
            subprocess.run(cmd, check=True)
            print(f"{method} completed successfully.")
            results[method] = True
        except subprocess.CalledProcessError as e:
            print(f"Error running {method}: {e}")
            results[method] = False

    # Print summary
    print("\nMethod execution summary:")
    for method, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {method}: {status}")

    # Generate visualization commands
    print("\nTo visualize the results, run the following commands:")
    for method in methods:
        if results[method]:
            if method == "block_matching":
                result_path = f"{args.output_dir}/block_matching_w{args.window_size}_o{args.overlap}/flow_results.mat"
                display_name = f"Block Matching (w={args.window_size}, o={args.overlap})"
            elif method == "farneback":
                result_path = f"{args.output_dir}/farneback_w{args.window_size}_i{args.iterations}_n{args.poly_n}_s{args.poly_sigma}_p{args.pyr_levels}/flow_results.mat"
                display_name = f"Farneback (w={args.window_size}, i={args.iterations})"
            elif method == "farneback_liushen":
                result_path = f"{args.output_dir}/farneback_liushen_w{args.window_size}_i{args.iterations}_n{args.poly_n}_s{args.poly_sigma}_p{args.pyr_levels}_a{args.liu_shen_alpha}/flow_results.mat"
                display_name = f"Farneback with Liu-Shen (w={args.window_size}, a={args.liu_shen_alpha})"
            elif method == "openpiv_smoothed":
                result_path = f"{args.output_dir}/openpiv_smoothed_w{args.window_size}_o{args.overlap}_s{args.smooth_sigma}/flow_results.mat"
                display_name = f"OpenPIV Smoothed (w={args.window_size}, o={args.overlap})"
            elif method == "dense_lucas_kanade":
                result_path = f"{args.output_dir}/dense_lucas_kanade_w{args.window_size}_i{args.iterations}_p{args.pyr_levels}/flow_results.mat"
                display_name = f"Dense Lucas-Kanade (w={args.window_size}, i={args.iterations})"
            elif method == "horn_schunck":
                result_path = f"{args.output_dir}/horn_schunck_a{args.liu_shen_alpha}_i{args.iterations}/flow_results.mat"
                display_name = f"Horn-Schunck (a={args.liu_shen_alpha}, i={args.iterations})"
            else:
                continue

            print(f"  python unified_display_results.py --all --input {result_path} --method \"{display_name}\"")

if __name__ == "__main__":
    main()
