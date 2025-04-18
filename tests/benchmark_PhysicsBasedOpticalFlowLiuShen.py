"""
Benchmark script to compare the performance of the original and Numba-optimized
implementations of the Liu-Shen physics-based optical flow algorithm.
"""

import os
import sys
import time
import numpy as np
from skimage.io import imread

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import both implementations
import PhysicsBasedOpticalFlowLiuShen
import PhysicsBasedOpticalFlowLiuShen_numba

def benchmark_implementations():
    """Benchmark both implementations on a real test case."""
    # Load test images
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testImages', 'Bits08', 'Ni06'))
    im1_path = os.path.join(test_dir, 'parabolic01_0.tif')
    im2_path = os.path.join(test_dir, 'parabolic01_1.tif')

    # Check if test images exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        print("Test images not found")
        return

    # Load images
    im1 = imread(im1_path).astype(np.float32)
    im2 = imread(im2_path).astype(np.float32)

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Set parameters
    h = 0.1  # Regularization parameter
    max_iterations = 30  # Limit iterations for benchmarking

    # Benchmark original implementation
    print("Benchmarking original implementation...")
    start_time = time.time()

    # Create a modified version of the function with limited iterations
    def original_func(im1, im2, h, U, V):
        # Override maxnum in the function body
        maxnum = max_iterations
        tol = 1e-8
        dx = 1
        dt = 1
        f = 0

        # Rest of the function is the same as the original
        im1 = im1/np.max(im1)
        im2 = im2/np.max(im2)

        from scipy.ndimage import convolve as filter2

        D  = np.array([[0, -1,  0], [0,  0,  0], [ 0, 1, 0] ], dtype=np.float32)/2
        M  = np.array([[1,  0, -1], [0,  0,  0], [-1, 0, 1] ], dtype=np.float32)/4
        F  = np.array([[0,  1,  0], [0,  0,  0], [ 0, 1, 0] ], dtype=np.float32)
        D2 = np.array([[0,  1,  0], [0, -2,  0], [ 0, 1, 0] ], dtype=np.float32)
        H  = np.array([[1,  1,  1], [1,  0,  1], [ 1, 1, 1] ], dtype=np.float32)

        D=np.flipud(np.fliplr(D))
        M=np.flipud(np.fliplr(M))
        F=np.flipud(np.fliplr(F))
        D2=np.flipud(np.fliplr(D2))
        H=np.flipud(np.fliplr(H))

        IIx = im1*filter2(im1, D/dx, mode='nearest')
        IIy = im1*filter2(im1, D.transpose()/dx, mode='nearest')
        II  = im1*im1
        Ixt = im1*filter2((im2-im1)/dt-f, D/dx, mode='nearest')
        Iyt = im1*filter2((im2-im1)/dt-f, D.transpose()/dx, mode='nearest')

        k=0
        total_error=100000000
        u=np.float32(U)
        v=np.float32(V)

        r,c=im2.shape

        B11, B12, B22 = PhysicsBasedOpticalFlowLiuShen.generate_invmatrix(im1, h, dx)

        error=0
        while total_error > tol and k < maxnum:
            bu = 2*IIx*filter2(u, D/dx, mode='nearest') + IIx*filter2(v, D.transpose()/dx, mode='nearest') + \
                   IIy*filter2(v, D/dx, mode='nearest') + II*filter2(u, F/(dx*dx), mode='nearest') + \
                   II*filter2(v, M/(dx*dx), mode='nearest') + h*filter2(u, H/(dx*dx), mode='constant')+Ixt

            bv = IIy*filter2(u, D/dx, mode='nearest') + IIx*filter2(u, D.transpose()/dx, mode='nearest') + \
                2*IIy*filter2(v, D.transpose()/dx, mode='nearest') + II*filter2(u, M/(dx*dx), mode='nearest') + \
                II*filter2(v, F.transpose()/(dx*dx), mode='nearest') + h*filter2(v, H/(dx*dx), mode='constant')+Iyt

            unew, vnew, total_error = PhysicsBasedOpticalFlowLiuShen.helper(B11, B12, B22, bu, bv, u, v, r, c)
            print('Iteration: ' + str(k) + ' - Total error: ' + str(total_error))

            u = unew
            v = vnew
            error=total_error
            k=k+1

        return u, v, error

    # Run original implementation
    u_orig, v_orig, err_orig = original_func(im1, im2, h, U, V)

    original_time = time.time() - start_time
    print(f"Original implementation took {original_time:.2f} seconds")

    # Benchmark Numba implementation
    print("\nBenchmarking Numba implementation...")
    start_time = time.time()

    # Create a modified version of the function with limited iterations
    def numba_func(im1, im2, h, U, V):
        # Override maxnum in the function body
        print(f"Computing Liu-Shen physics-based optical flow with h={h}")
        print(f"Image size: {im1.shape}")

        # Initialize parameters with our custom maxnum
        f = 0  # Boundary assumption
        maxnum = max_iterations  # Maximum number of iterations
        tol = 1e-8  # Convergence tolerance
        dx = 1  # Spatial step
        dt = 1  # Time step

        # Normalize images
        im1 = im1 / np.max(im1)
        im2 = im2 / np.max(im2)

        from scipy.ndimage import convolve as filter2

        # Define kernels
        D = np.array([
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32) / 2  # partial derivative

        M = np.array([
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]
        ], dtype=np.float32) / 4  # mixed partial derivatives

        F = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)  # average

        D2 = np.array([
            [0, 1, 0],
            [0, -2, 0],
            [0, 1, 0]
        ], dtype=np.float32)  # partial derivative

        H = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.float32)

        # MATLAB imfilter employs correlation, Python conv2 uses convolution, so we must mirror the kernels
        D = np.flipud(np.fliplr(D))
        M = np.flipud(np.fliplr(M))
        F = np.flipud(np.fliplr(F))
        D2 = np.flipud(np.fliplr(D2))
        H = np.flipud(np.fliplr(H))

        # Compute derivatives
        IIx = im1 * filter2(im1, D / dx, mode='nearest')
        IIy = im1 * filter2(im1, D.transpose() / dx, mode='nearest')
        II = im1 * im1
        Ixt = im1 * filter2((im2 - im1) / dt - f, D / dx, mode='nearest')
        Iyt = im1 * filter2((im2 - im1) / dt - f, D.transpose() / dx, mode='nearest')

        # Initialize variables
        k = 0
        total_error = 100000000
        u = np.float32(U)
        v = np.float32(V)
        r, c = im2.shape

        # Generate inverse matrix
        B11, B12, B22 = PhysicsBasedOpticalFlowLiuShen_numba.generate_invmatrix(im1, h, dx)

        # Iterative refinement
        error = 0
        print("Liu-Shen iterations:")
        while total_error > tol and k < maxnum:
            # Compute bu and bv
            bu = 2 * IIx * filter2(u, D / dx, mode='nearest') + \
                 IIx * filter2(v, D.transpose() / dx, mode='nearest') + \
                 IIy * filter2(v, D / dx, mode='nearest') + \
                 II * filter2(u, F / (dx * dx), mode='nearest') + \
                 II * filter2(v, M / (dx * dx), mode='nearest') + \
                 h * filter2(u, H / (dx * dx), mode='constant') + Ixt

            bv = IIy * filter2(u, D / dx, mode='nearest') + \
                 IIx * filter2(u, D.transpose() / dx, mode='nearest') + \
                 2 * IIy * filter2(v, D.transpose() / dx, mode='nearest') + \
                 II * filter2(u, M / (dx * dx), mode='nearest') + \
                 II * filter2(v, F.transpose() / (dx * dx), mode='nearest') + \
                 h * filter2(v, H / (dx * dx), mode='constant') + Iyt

            # Compute flow update
            unew, vnew = PhysicsBasedOpticalFlowLiuShen_numba._compute_flow_update(B11, B12, B22, bu, bv)

            # Compute error
            total_error = PhysicsBasedOpticalFlowLiuShen_numba._compute_total_error(unew, vnew, u, v, r, c)

            # Update flow
            u = unew
            v = vnew
            error = total_error
            k += 1

            print(f"Iteration: {k} - Total error: {total_error}")

        print(f"Liu-Shen completed after {k} iterations with error {error:.6f}")
        return u, v, error

    # Run Numba implementation
    u_numba, v_numba, err_numba = numba_func(im1, im2, h, U, V)

    numba_time = time.time() - start_time
    print(f"Numba implementation took {numba_time:.2f} seconds")

    # Compare performance
    speedup = original_time / numba_time
    print(f"\nSpeedup: {speedup:.2f}x")

    # Verify that the results are the same
    u_diff = np.abs(u_orig - u_numba).mean()
    v_diff = np.abs(v_orig - v_numba).mean()
    print(f"Mean absolute difference in U: {u_diff:.6f}")
    print(f"Mean absolute difference in V: {v_diff:.6f}")

    return speedup

if __name__ == "__main__":
    benchmark_implementations()
