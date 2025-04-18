#!/usr/bin/env python
"""
Optimized Numba implementation of the Horn-Schunck optical flow algorithm.

This implementation uses Numba for CPU acceleration and follows the same algorithm
as the original implementation but with better performance.
"""

from __future__ import division
import numpy as np
from numba import njit, prange
from scipy.ndimage import convolve as filter2
from scipy.linalg import norm
from matplotlib import pyplot as plt
from tqdm import tqdm

QUIVER = 5

class HSOpticalFlowAlgoAdapter(object):
    def __init__(self, alphas, Niter, provideGenericPyramidalDefaults=True):
        self.provideGenericPyramidalDefaults = provideGenericPyramidalDefaults
        self.alphas = alphas
        self.Niter = Niter

    def compute(self, im1, im2, U, V):
        alpha = self.alphas.pop() #Remove last alpha from the list
        return HS(im1, im2, alpha, self.Niter, U, V)

    def getAlgoName(self):
        return 'Horn-Schunck'

    def hasGenericPyramidalDefaults(self):
        return self.provideGenericPyramidalDefaults

    def getGenericPyramidalDefaults(self):
        parameters = {}
        parameters['warping'] = True
        parameters['biLinear'] = True
        parameters['scaling'] = True
        return parameters

@njit(parallel=True)
def _compute_derivatives(im1, im2):
    """Compute image derivatives using Numba acceleration."""
    height, width = im1.shape
    fx = np.zeros_like(im1)
    fy = np.zeros_like(im1)
    ft = np.zeros_like(im1)
    
    # Define kernels
    kernelX = np.array([[-1, 1], [-1, 1]], dtype=np.float32) * 0.25
    kernelY = np.array([[-1, -1], [1, 1]], dtype=np.float32) * 0.25
    kernelT = np.ones((2, 2), dtype=np.float32) * 0.25
    
    # Compute derivatives
    for y in prange(1, height - 1):
        for x in range(1, width - 1):
            # X derivative
            fx[y, x] = (im1[y-1, x+1] + im1[y+1, x+1] - im1[y-1, x-1] - im1[y+1, x-1]) * 0.25 + \
                       (im2[y-1, x+1] + im2[y+1, x+1] - im2[y-1, x-1] - im2[y+1, x-1]) * 0.25
            
            # Y derivative
            fy[y, x] = (im1[y+1, x-1] + im1[y+1, x+1] - im1[y-1, x-1] - im1[y-1, x+1]) * 0.25 + \
                       (im2[y+1, x-1] + im2[y+1, x+1] - im2[y-1, x-1] - im2[y-1, x+1]) * 0.25
            
            # T derivative
            ft[y, x] = (im2[y-1, x-1] + im2[y-1, x] + im2[y, x-1] + im2[y, x]) * 0.25 - \
                       (im1[y-1, x-1] + im1[y-1, x] + im1[y, x-1] + im1[y, x]) * 0.25
    
    return fx, fy, ft

@njit(parallel=True)
def _compute_flow_update(fx, fy, ft, u_avg, v_avg, alpha):
    """Compute flow update using Numba acceleration."""
    height, width = fx.shape
    u_new = np.zeros_like(u_avg)
    v_new = np.zeros_like(v_avg)
    
    for y in prange(1, height - 1):
        for x in range(1, width - 1):
            # Common part of update step
            denominator = alpha**2 + fx[y, x]**2 + fy[y, x]**2
            if denominator > 1e-9:
                der = (fx[y, x] * u_avg[y, x] + fy[y, x] * v_avg[y, x] + ft[y, x]) / denominator
                
                # Iterative step
                u_new[y, x] = u_avg[y, x] - fx[y, x] * der
                v_new[y, x] = v_avg[y, x] - fy[y, x] * der
            else:
                u_new[y, x] = u_avg[y, x]
                v_new[y, x] = v_avg[y, x]
    
    return u_new, v_new

@njit
def _compute_local_averages(U, V, kernel):
    """Compute local averages of flow vectors."""
    height, width = U.shape
    u_avg = np.zeros_like(U)
    v_avg = np.zeros_like(V)
    
    # Apply kernel to compute local averages
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            u_sum = 0.0
            v_sum = 0.0
            weight_sum = 0.0
            
            for j in range(-1, 2):
                for i in range(-1, 2):
                    weight = kernel[j+1, i+1]
                    u_sum += U[y+j, x+i] * weight
                    v_sum += V[y+j, x+i] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                u_avg[y, x] = u_sum
                v_avg[y, x] = v_sum
    
    return u_avg, v_avg

def HS_helper(alpha, Niter, kernel, U, V, fx, fy, ft):
    """Helper function for Horn-Schunck algorithm."""
    # Use tqdm for progress tracking
    with tqdm(total=Niter, desc="Horn-Schunck iterations") as pbar:
        for _ in range(Niter):
            # Compute local averages of the flow vectors
            u_avg, v_avg = _compute_local_averages(U, V, kernel)
            
            # Update flow
            U, V = _compute_flow_update(fx, fy, ft, u_avg, v_avg, alpha)
            
            # Update progress bar
            pbar.update(1)
    
    return U, V

def HS(im2, im1, alpha, Niter, U, V):
    """
    Horn-Schunck optical flow algorithm.
    
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    U: initial x-component velocity vector
    V: initial y-component velocity vector
    """
    print(f"Computing Horn-Schunck optical flow with alpha={alpha}, iterations={Niter}")
    print(f"Image size: {im1.shape}")

    # Estimate derivatives
    fx, fy, ft = _compute_derivatives(im1, im2)

    # Averaging kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,    0, 1/6],
                       [1/12, 1/6, 1/12]], np.float32)

    if (np.size(fx, 0) > 100 and np.size(fx, 1) > 100):
        print(f"Sample derivatives at (100,100): fx={fx[100,100]:.4f}, fy={fy[100,100]:.4f}, ft={ft[100,100]:.4f}")

    # Initialize error
    total_error = 0.0
    
    # Iteration to reduce error
    U_new, V_new = HS_helper(alpha, Niter, kernel, np.float32(U), np.float32(V), fx, fy, ft)
    
    # Compute error
    total_error = (norm(U_new - U, 'fro') + norm(V_new - V, 'fro')) / (im1.shape[0] * im1.shape[1])
    print(f"Total error: {total_error:.6f}")

    return U_new, V_new, total_error

def compareGraphs(u, v, Inew, title="Horn-Schunck Flow", scale=3, output_file=None):
    """
    Plot optical flow as quiver plot.
    
    Args:
        u, v: Flow components
        Inew: Image to overlay flow on
        title: Plot title
        scale: Scale factor for arrows
        output_file: If provided, save plot to this file
    """
    ax = plt.figure(figsize=(10, 8)).gca()
    ax.imshow(Inew, cmap='gray')
    
    # Create quiver plot with downsampling
    y, x = np.mgrid[0:u.shape[0]:QUIVER, 0:u.shape[1]:QUIVER]
    u_skip = u[::QUIVER, ::QUIVER]
    v_skip = v[::QUIVER, ::QUIVER]
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(u_skip**2 + v_skip**2)
    
    # Plot quiver with colors based on magnitude
    quiv = ax.quiver(x, y, u_skip, v_skip, magnitude, 
                    scale=50, scale_units='inches',
                    cmap='jet', clim=[0, np.percentile(magnitude, 95)])
    plt.colorbar(quiv, ax=ax, label='Magnitude (pixels/frame)')
    
    ax.set_title(title)
    ax.set_xlim(0, u.shape[1])
    ax.set_ylim(u.shape[0], 0)  # Invert y-axis to match image coordinates
    
    if output_file:
        plt.savefig(output_file, dpi=200)
    
    plt.draw()
    plt.pause(0.01)

def demo(_):
    """Demo function (unused)"""
    # Parameter is unused
    return

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='Numba-accelerated Horn Schunck Optical Flow')
    p.add_argument('stem', help='path/stem of files to analyze')
    p = p.parse_args()
