"""
MIT LicenseCopyright (c) [2021-2024] [Luís Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:The above copyright notice and
this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/env python
"""
Optimized Numba implementation of the Liu-Shen physics-based optical flow algorithm.

This implementation uses Numba for CPU acceleration and follows the same algorithm
as the original implementation but with better performance.
"""

import numpy as np
from numba import njit, prange
from scipy.ndimage import convolve as filter2
from tqdm import tqdm

class LiuShenOpticalFlowAlgoAdapter(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def compute(self, im1, im2, U, V):
        [resV, resU, error] = physicsBasedOpticalFlowLiuShen(im1, im2, self.alpha, V, U)
        return [resU, resV, error]

    def getAlgoName(self):
        return 'Liu-Shen Physics based OF'

    def hasGenericPyramidalDefaults(self):
        return False

def _compute_determinant_and_inverse(A11, A22, A12):
    """Compute determinant and inverse of matrix A."""
    DetA = A11 * A22 - A12 * A12

    # Avoid division by zero
    DetA = np.maximum(DetA, 1e-10)

    B11 = A22 / DetA
    B12 = -A12 / DetA
    B22 = A11 / DetA

    return B11, B12, B22

def _compute_flow_update(B11, B12, B22, bu, bv):
    """Compute flow update using matrix B and vectors bu, bv."""
    unew = -(B11 * bu + B12 * bv)
    vnew = -(B12 * bu + B22 * bv)

    return unew, vnew

@njit
def _compute_total_error(unew, vnew, u, v, r, c):
    """Compute total error between old and new flow fields."""
    u_diff = unew - u
    v_diff = vnew - v

    u_norm = np.sum(u_diff * u_diff)
    v_norm = np.sum(v_diff * v_diff)

    total_error = np.sqrt(u_norm + v_norm) / (r * c)

    return total_error

def generate_invmatrix(im, h, dx):
    """Generate inverse matrix for the Liu-Shen algorithm."""
    # Define kernels
    M = np.array([
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1]
    ], dtype=np.float32) / 4  # mixed partial derivatives

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
    M = np.flipud(np.fliplr(M))
    D2 = np.flipud(np.fliplr(D2))
    H = np.flipud(np.fliplr(H))

    h = np.float32(h)

    # Compute matrix components
    cmtx = filter2(np.ones(im.shape, dtype=np.float32), H / (dx * dx), mode='constant')
    A11 = im * (filter2(im, D2 / (dx * dx), mode='nearest') - 2 * im / (dx * dx)) - h * cmtx
    A22 = im * (filter2(im, D2.transpose() / (dx * dx), mode='nearest') - 2 * im / (dx * dx)) - h * cmtx
    A12 = im * filter2(im, M / (dx * dx), mode='nearest')

    # Compute inverse matrix
    B11, B12, B22 = _compute_determinant_and_inverse(A11, A22, A12)

    return B11, B12, B22

def physicsBasedOpticalFlowLiuShen(im1, im2, h, U, V):
    """
    Liu-Shen physics-based optical flow algorithm.

    Args:
        im1: First image
        im2: Second image
        h: Regularization parameter
        U: Initial horizontal flow component
        V: Initial vertical flow component

    Returns:
        u, v: Updated flow components
        error: Final error
    """
    print(f"Computing Liu-Shen physics-based optical flow with h={h}")
    print(f"Image size: {im1.shape}")

    # Initialize parameters
    f = 0  # Boundary assumption
    maxnum = 60  # Maximum number of iterations
    tol = 1e-8  # Convergence tolerance
    dx = 1  # Spatial step
    dt = 1  # Time step

    # Normalize images
    im1 = im1 / np.max(im1)
    im2 = im2 / np.max(im2)

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
    B11, B12, B22 = generate_invmatrix(im1, h, dx)

    # Iterative refinement
    error = 0
    with tqdm(total=maxnum, desc="Liu-Shen iterations") as pbar:
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
            unew, vnew = _compute_flow_update(B11, B12, B22, bu, bv)

            # Compute error
            total_error = _compute_total_error(unew, vnew, u, v, r, c)

            # Update flow
            u = unew
            v = vnew
            error = total_error
            k += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"error": f"{error:.6f}"})

    print(f"Liu-Shen completed after {k} iterations with error {error:.6f}")
    return u, v, error
