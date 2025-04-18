#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numba-accelerated implementation of the Farneback optical flow algorithm.

This is a CPU-based implementation that uses Numba for acceleration instead of OpenCL.
It follows the same algorithm as the OpenCL version but is designed to work without GPU hardware.

Based on the original OpenCV implementation:
Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
"""

import os
import numpy as np
from numba import njit, prange, float32, int32
import PIL
from PIL import Image
from scipy.ndimage import gaussian_filter
from GaussianKernelBitExact import getGaussianKernelBitExact
from tqdm import tqdm

def imresize(im, res):
    """Resize an image using PIL's BILINEAR interpolation."""
    return np.array(Image.fromarray(im).resize(res, PIL.Image.BILINEAR))

@njit(cache=True)
def _bilinear_interpolate(img, x, y):
    """Bilinear interpolation for float coordinates."""
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Get weights
    wx = x - x0
    wy = y - y0

    # Get pixel values with bounds checking
    v00 = img[y0, x0] if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1] else 0
    v01 = img[y0, x1] if 0 <= y0 < img.shape[0] and 0 <= x1 < img.shape[1] else 0
    v10 = img[y1, x0] if 0 <= y1 < img.shape[0] and 0 <= x0 < img.shape[1] else 0
    v11 = img[y1, x1] if 0 <= y1 < img.shape[0] and 0 <= x1 < img.shape[1] else 0

    # Interpolate
    return (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v01 + (1 - wx) * wy * v10 + wx * wy * v11

@njit(cache=True)
def _polynomial_expansion_kernel(src, dst, height, width, n, g):
    """Kernel function for polynomial expansion."""
    for y in prange(n, height - n):
        for x in range(n, width - n):
            # Compute polynomial expansion coefficients
            b1 = 0.0
            b2 = 0.0
            b3 = 0.0
            b4 = 0.0
            b5 = 0.0
            b6 = 0.0

            # Apply horizontal and vertical filters
            for j in range(-n, n+1):
                for i in range(-n, n+1):
                    val = src[y+j, x+i]
                    gval = g[abs(i)] * g[abs(j)]
                    b1 += val * gval
                    b2 += val * gval * i
                    b3 += val * gval * j
                    b4 += val * gval * i * i
                    b5 += val * gval * i * j
                    b6 += val * gval * j * j

            # Store coefficients
            dst[5*y, x] = b1
            dst[5*y+1, x] = b2
            dst[5*y+2, x] = b3
            dst[5*y+3, x] = b4
            dst[5*y+4, x] = b5
            # We don't store b6 as it's not needed for the algorithm

    return dst

@njit(cache=True)
def _update_matrices_kernel(flowX, flowY, RA, RB, M, height, width):
    """Kernel function for updating matrices."""
    for y in prange(height):
        for x in range(width):
            # Get flow at current pixel
            fx = flowX[y, x]
            fy = flowY[y, x]

            # Compute coordinates in the second image
            x1 = x + fx
            y1 = y + fy

            # Skip if outside image boundaries
            if x1 < 0 or x1 >= width-1 or y1 < 0 or y1 >= height-1:
                continue

            # Bilinear interpolation weights
            w00 = (1.0 - (x1 - int(x1))) * (1.0 - (y1 - int(y1)))
            w01 = (1.0 - (x1 - int(x1))) * (y1 - int(y1))
            w10 = (x1 - int(x1)) * (1.0 - (y1 - int(y1)))
            w11 = (x1 - int(x1)) * (y1 - int(y1))

            # Integer coordinates
            x1_i = int(x1)
            y1_i = int(y1)

            # For each coefficient
            for k in range(5):
                idx = 5 * y + k
                idx1_00 = 5 * y1_i + k
                idx1_01 = 5 * (y1_i + 1) + k

                # Bilinear interpolation of RB coefficients
                rb = (RB[idx1_00, x1_i] * w00 +
                      RB[idx1_01, x1_i] * w01 +
                      RB[idx1_00, x1_i + 1] * w10 +
                      RB[idx1_01, x1_i + 1] * w11)

                # Compute difference for matrix M
                M[idx, x] = RA[idx, x] - rb

    return M

@njit(cache=True)
def _update_flow_kernel(M, flowX, flowY, height, width):
    """Kernel function for updating flow."""
    for y in prange(height):
        for x in range(width):
            # Get coefficients from M
            b1 = M[5*y, x]
            b2 = M[5*y+1, x]
            b3 = M[5*y+2, x]
            b4 = M[5*y+3, x]
            b5 = M[5*y+4, x]
            # We're not using b6 from M[5*y+4+height, x] anymore
            # Instead, we'll compute it based on the other coefficients
            b6 = b4  # Approximation: use b4 as b6 (both are second derivatives)

            # Compute determinant
            det = (b4*b6 - b5*b5)

            # Skip if determinant is too small
            if abs(det) < 1e-9:
                continue

            # Compute flow update
            inv_det = 1.0 / det
            flowX[y, x] = (b6*b2 - b5*b3) * inv_det
            flowY[y, x] = (b4*b3 - b5*b2) * inv_det

    return flowX, flowY

@njit(cache=True)
def _box_filter5_kernel(src, dst, height, width, radius):
    """Kernel function for box filter on 5-channel data."""
    for y in prange(height):
        for x in range(width):
            # For each channel
            for k in range(5):
                sum_val = 0.0
                count = 0

                # Apply box filter
                for j in range(-radius, radius+1):
                    for i in range(-radius, radius+1):
                        nx = x + i
                        ny = y + j

                        # Check boundaries
                        if nx >= 0 and nx < width and ny >= 0 and ny < height:
                            sum_val += src[5*ny+k, nx]
                            count += 1

                # Normalize
                if count > 0:
                    dst[5*y+k, x] = sum_val / count

    return dst

@njit(cache=True)
def _gaussian_blur5_horizontal_kernel(src, temp, height, width, kernel, radius):
    """Horizontal pass of Gaussian blur on 5-channel data."""
    for y in prange(height):
        for k in range(5):
            for x in range(width):
                sum_val = 0.0
                wsum = 0.0

                for i in range(-radius, radius+1):
                    nx = x + i
                    if nx >= 0 and nx < width:
                        w = kernel[abs(i)]
                        sum_val += src[5*y+k, nx] * w
                        wsum += w

                if wsum > 0:
                    temp[5*y+k, x] = sum_val / wsum

    return temp

@njit(cache=True)
def _gaussian_blur5_vertical_kernel(temp, dst, height, width, kernel, radius):
    """Vertical pass of Gaussian blur on 5-channel data."""
    for x in prange(width):
        for k in range(5):
            for y in range(height):
                sum_val = 0.0
                wsum = 0.0

                for j in range(-radius, radius+1):
                    ny = y + j
                    if ny >= 0 and ny < height:
                        w = kernel[abs(j)]
                        sum_val += temp[5*ny+k, x] * w
                        wsum += w

                if wsum > 0:
                    dst[5*y+k, x] = sum_val / wsum

    return dst

class Farneback_Numba(object):
    """
    Provides a Numba-accelerated implementation of the Farneback optical flow algorithm.
    It also implements the GenericPyramidalOpticalFlow client algorithms interface.
    """
    def __init__(self, windowSize=33, Niters=5, polyN=7, polySigma=1.5, useGaussian=True, pyrScale=0.5, pyramidalLevels=1,
                 provideGenericPyramidalDefaults=True):
        assert pyramidalLevels >= 1, 'Pyramidal levels must be greater or equal than 1'
        self.windowSize = windowSize
        self.numIters = Niters
        self.polyN = int(polyN)
        self.polySigma = polySigma
        self.useGaussianFilter = useGaussian
        self.pyramidalLevels = pyramidalLevels-1
        self.pyrScale = pyrScale
        self.provideGenericPyramidalDefaults = provideGenericPyramidalDefaults

        if windowSize & 1 == 0:
            raise Exception('windowSize must be an odd value')

        # Initialize polynomial expansion constants
        self.setPolynomialExpansionConsts()

    def FarnebackPrepareGaussian(self):
        """Prepare Gaussian filter coefficients for polynomial expansion."""
        n = self.polyN
        sigma = self.polySigma

        if sigma < 1.19209289550781250000000000000000000e-7:
            sigma = n*0.3

        g = np.zeros([2*n+1], dtype=np.float32)
        xg = np.zeros([2*n+1], dtype=np.float32)
        xxg = np.zeros([2*n+1], dtype=np.float32)

        s = np.float64(0.0)
        for x in range(-n, n+1):
            g[x + n] = np.exp(-x*x/(2*sigma*sigma))
            s += g[x + n]

        s = 1.0/s
        for x in range(-n, n+1):
            g[x + n] = np.float32(g[x + n]*s)
            xg[x + n] = np.float32(x*g[x + n])
            xxg[x + n] = np.float32(x*x*g[x + n])

        G = np.zeros((6, 6), np.float64)
        for y in range(-n, n+1):
            for x in range(-n, n+1):
                G[0,0] += g[y + n]*g[x + n]
                G[1,1] += g[y + n]*g[x + n]*x*x
                G[3,3] += g[y + n]*g[x + n]*x*x*x*x
                G[5,5] += g[y + n]*g[x + n]*x*x*y*y

        G[2,2] = G[0,3] = G[0,4] = G[3,0] = G[4,0] = G[1,1]
        G[4,4] = G[3,3]
        G[3,4] = G[4,3] = G[5,5]

        invG = np.linalg.inv(G)

        ig11 = invG[1,1]
        ig03 = invG[0,3]
        ig33 = invG[3,3]
        ig55 = invG[5,5]

        return g, xg, xxg, ig11, ig03, ig33, ig55

    def setPolynomialExpansionConsts(self):
        """Set constants for polynomial expansion."""
        n = self.polyN
        m_igd = np.zeros(4, dtype=np.float64)
        m_ig = np.zeros(4, dtype=np.float32)
        g, xg, xxg, m_igd[0], m_igd[1], m_igd[2], m_igd[3] = self.FarnebackPrepareGaussian()

        t_g = g[n:].reshape([1, n+1])
        t_xg = xg[n:].reshape([1, n+1])
        t_xxg = xxg[n:].reshape([1, n+1])

        m_g = t_g.copy()
        m_xg = t_xg.copy()
        m_xxg = t_xxg.copy()

        m_ig[0] = np.float32(m_igd[0])
        m_ig[1] = np.float32(m_igd[1])
        m_ig[2] = np.float32(m_igd[2])
        m_ig[3] = np.float32(m_igd[3])

        self.matrixG = m_g
        self.matrixXG = m_xg
        self.matrixXXG = m_xxg
        self.matrixIG = m_ig
        self.matrixIGD = m_igd

    def getGaussianKernel(self, n, sigma):
        """Get Gaussian kernel for blurring."""
        _, kernel_bitexact = getGaussianKernelBitExact(n, sigma)
        return np.float32(kernel_bitexact)

    def polynomialExpansion(self, src, dst):
        """
        Compute polynomial expansion for the source image using Numba acceleration.
        """
        height, width = src.shape
        n = self.polyN
        g = self.matrixG[0]
        ig = self.matrixIG

        # Use Numba-accelerated kernel
        dst = _polynomial_expansion_kernel(src, dst, height, width, n, g)

        # Apply inverse G matrix (not parallelized to avoid race conditions)
        for y in range(n, height - n):
            for x in range(n, width - n):
                dst[5*y+1, x] *= ig[0]
                dst[5*y+2, x] *= ig[0]
                dst[5*y+3, x] = (dst[5*y+3, x] - dst[5*y, x] * ig[1] / ig[0]) * ig[2]
                dst[5*y+4, x] *= ig[3]

        return dst

    def updateMatrices(self, flowX, flowY, RA, RB, M):
        """
        Update matrices based on current flow using Numba acceleration.
        """
        height = flowX.shape[0]
        width = flowX.shape[1]

        # Use Numba-accelerated kernel
        M = _update_matrices_kernel(flowX, flowY, RA, RB, M, height, width)

        return M

    def updateFlow(self, M, flowX, flowY):
        """
        Update flow based on coefficient matrices using Numba acceleration.
        """
        height = flowX.shape[0]
        width = flowX.shape[1]

        # Use Numba-accelerated kernel
        flowX, flowY = _update_flow_kernel(M, flowX, flowY, height, width)

        return flowX, flowY

    def boxFilter5(self, src, ksize, dst):
        """
        Apply box filter to 5-channel data using Numba acceleration.
        """
        height = src.shape[0] // 5
        width = src.shape[1]
        radius = ksize // 2

        # Use Numba-accelerated kernel
        dst = _box_filter5_kernel(src, dst, height, width, radius)

        return dst

    def gaussianBlur5(self, src, ksize, dst):
        """
        Apply Gaussian blur to 5-channel data using Numba acceleration.
        """
        height = src.shape[0] // 5
        width = src.shape[1]
        radius = ksize // 2

        # Get Gaussian kernel
        kernel = self.getGaussianKernel(2*radius+1, radius)

        # Temporary buffer for separable filtering
        temp = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)

        # Use Numba-accelerated kernels for separable filtering
        temp = _gaussian_blur5_horizontal_kernel(src, temp, height, width, kernel, radius)
        dst = _gaussian_blur5_vertical_kernel(temp, dst, height, width, kernel, radius)

        return dst

    def updateFlowBoxFilter(self, RA, RB, flowX, flowY, M, bufM, blockSize, updateMatrices):
        """Update flow using box filter."""
        try:
            bufM = self.boxFilter5(M, blockSize//2, bufM)
        except Exception as e:
            raise Exception('Failed to compute boxFilter5 in updateFlowBoxFilter', e)

        # swap(M, bufM)
        M, bufM = bufM, M
        try:
            flowX, flowY = self.updateFlow(M, flowX, flowY)
        except Exception as e:
            raise Exception('Failed to updateFlow in updateFlowBoxFilter', e)

        if updateMatrices:
            M = self.updateMatrices(flowX, flowY, RA, RB, M)

        return flowX, flowY, M, bufM

    def updateFlowGaussianBlur(self, RA, RB, flowX, flowY, M, bufM, blockSize, updateMatrices):
        """Update flow using Gaussian blur."""
        try:
            bufM = self.gaussianBlur5(M, blockSize//2, bufM)
        except Exception as e:
            raise Exception('Failed to compute gaussianBlur5 in updateFlowGaussianBlur', e)

        # swap(M, bufM)
        M, bufM = bufM, M
        try:
            flowX, flowY = self.updateFlow(M, flowX, flowY)
        except Exception as e:
            raise Exception('Failed to updateFlow in updateFlowGaussianBlur', e)

        if updateMatrices:
            M = self.updateMatrices(flowX, flowY, RA, RB, M)

        return flowX, flowY, M, bufM

    def compute(self, im1, im2, U, V):
        """
        Compute optical flow between two images.

        Args:
            im1: First image
            im2: Second image
            U: Initial horizontal flow component
            V: Initial vertical flow component

        Returns:
            U, V: Updated flow components
            error: Error message
        """
        assert(self.polyN == 5 or self.polyN == 7)
        assert(im1.shape == im2.shape and self.pyrScale < 1)
        assert(U.shape == im1.shape and V.shape == im1.shape)

        print(f"\nStarting Farneback optical flow computation")
        print(f"Image size: {im1.shape}")
        print(f"Parameters: polyN={self.polyN}, polySigma={self.polySigma}, windowSize={self.windowSize}, iterations={self.numIters}, pyramidalLevels={self.pyramidalLevels+1}")

        min_size = 32
        size = im1.shape  # [0] - height, [1] - width
        prevFlowX = None
        prevFlowY = None
        curFlowX = None
        curFlowY = None

        flowX0 = U
        flowY0 = V

        # Crop unnecessary pyramidal levels
        scale = 1
        finalNumLevels = 0
        while finalNumLevels < self.pyramidalLevels:
            scale *= self.pyrScale
            if (size[1]*scale < min_size or size[0]*scale < min_size):
                break
            finalNumLevels += 1

        for k in np.arange(finalNumLevels, -1, -1):
            print(f"\nProcessing pyramid level {k} of {finalNumLevels}")

            scale = 1.0
            for i in np.arange(0, k):
                scale *= self.pyrScale

            sigma = (1.0/scale - 1.0) * 0.5
            smoothSize = int(round(sigma*5)) | 1
            smoothSize = max(smoothSize, 3)

            width = int(round(size[1] * scale))
            height = int(round(size[0] * scale))

            print(f"Level {k} image size: {height}x{width}, scale: {scale:.3f}, sigma: {sigma:.3f}")

            if prevFlowX is None:
                curFlowX = imresize(flowX0, (width, height))
                curFlowY = imresize(flowY0, (width, height))
                curFlowX *= scale
                curFlowY *= scale
            else:
                curFlowX = imresize(prevFlowX, (width, height))
                curFlowY = imresize(prevFlowY, (width, height))
                curFlowX *= 1.0/self.pyrScale
                curFlowY *= 1.0/self.pyrScale

            M = np.zeros((5*height, width), np.float32)
            bufM = np.zeros((5*height, width), np.float32)
            RA = np.zeros((5*height, width), np.float32)
            RB = np.zeros((5*height, width), np.float32)

            # Prepare images for current pyramid level
            blurredFrameA = gaussian_filter(im1, sigma)
            blurredFrameB = gaussian_filter(im2, sigma)

            pyrLevelA = imresize(blurredFrameA, (width, height))
            pyrLevelB = imresize(blurredFrameB, (width, height))

            # Polynomial expansion
            print("Computing polynomial expansion for frame A...")
            try:
                RA = self.polynomialExpansion(pyrLevelA, RA)
            except Exception as e:
                raise Exception(f'Failed to do polynomial expansion for frame A, level {k}', e)

            print("Computing polynomial expansion for frame B...")
            try:
                RB = self.polynomialExpansion(pyrLevelB, RB)
            except Exception as e:
                raise Exception(f'Failed to do polynomial expansion for frame B, level {k}', e)

            # Update matrices based on current flow
            print("Updating matrices based on current flow...")
            M = self.updateMatrices(curFlowX, curFlowY, RA, RB, M)

            # Iteratively update flow
            for i in np.arange(0, self.numIters):
                print(f"  Iteration {i+1} of {self.numIters}")
                if self.useGaussianFilter:
                    curFlowX, curFlowY, M, bufM = self.updateFlowGaussianBlur(
                        RA, RB, curFlowX, curFlowY, M, bufM, self.windowSize, i < self.numIters-1)
                else:
                    curFlowX, curFlowY, M, bufM = self.updateFlowBoxFilter(
                        RA, RB, curFlowX, curFlowY, M, bufM, self.windowSize, i < self.numIters-1)
                print(f"  Flow range: U={curFlowX.min():.2f} to {curFlowX.max():.2f}, V={curFlowY.min():.2f} to {curFlowY.max():.2f}")

            prevFlowX = curFlowX
            prevFlowY = curFlowY

        U = curFlowX
        V = curFlowY
        error = None

        return U, V, error

    def getAlgoName(self):
        """Get algorithm name."""
        return 'Farneback Numba'

    def hasGenericPyramidalDefaults(self):
        """Check if algorithm provides generic pyramidal defaults."""
        return self.provideGenericPyramidalDefaults

    def getGenericPyramidalDefaults(self):
        """Get generic pyramidal defaults."""
        parameters = {}
        parameters['warping'] = False
        parameters['scaling'] = True
        return parameters
