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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import numpy as np
from skimage.io import imread
import scipy.io
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow
from PhysicsBasedOpticalFlowLiuShen import LiuShenOpticalFlowAlgoAdapter

# Try to import OpenCL version first, fall back to Numba if it fails
try:
    from denseLucasKanade_PyCL import denseLucasKanade_PyCl
    use_opencl = False
    print("Using OpenCL implementation of Dense Lucas-Kanade")
except Exception as e:
    from denseLucasKanade_Numba import denseLucasKanade_Numba
    use_opencl = False
    print(f"OpenCL import failed: {e}")
    print("Falling back to Numba implementation of Dense Lucas-Kanade")

def save_flow(U, V, filename):
    margins = { 'top' : 0,
                'left' : 0,
                'bottom' : 0,
                'right' : 0 }
    results = { 'u' : U,
                'v' : V,
                'iaWidth' : 1,
                'iaHeight' : 1,
                'margins' : margins  }

    parameters = { 'overlapFactor' : 1.0,
                   'imageHeight' : np.size(U, 0),
                   'imageWidth' : np.size(U, 1) }

    scipy.io.savemat(filename, mdict={'velocities': results, 'parameters': parameters})

FILTER=2
FILTER_OPT=0.48
pyramidalLevels = 2  # Using exactly 2 pyramid levels as requested
kLevels = 1
useLiuShenOF = True  # Enable Liu-Shen enhancement
# Note: minWindowSize is not supported by genericPyramidalOpticalFlow
basePath=os.path.join('testImages','Bits08','Ni06')
fn1=os.path.join(basePath, 'parabolic01_0.tif')
fn2=os.path.join(basePath, 'parabolic01_1.tif')

print(fn1);

#Iold = imread(fn1,flatten=True).astype(np.float32)  #flatten=True is rgb2gray
#Inew = imread(fn2,flatten=True).astype(np.float32)
#Iold = imread(fn1,as_gray=True).astype(np.float32)  #as_gray=True only supported after Ubuntu 18.04
#Inew = imread(fn2,as_gray=True).astype(np.float32)
Iold = imread(fn1).astype(np.float32)
Inew = imread(fn2).astype(np.float32)

# Create the appropriate Lucas-Kanade adapter based on availability
if use_opencl:
    lkAdapter = denseLucasKanade_PyCl(Niter=5, halfWindow=13, platformID=0) #Python OpenCL direct implementation
else:
    lkAdapter = denseLucasKanade_Numba(Niter=5, halfWindow=13, provideGenericPyramidalDefaults=True) #Numba CPU implementation
if useLiuShenOF:
    lsAdapter = LiuShenOpticalFlowAlgoAdapter(0.1)  # Using 0.1 for alpha parameter
else:
    lsAdapter = None

[U,V] = genericPyramidalOpticalFlow(Iold, Inew, FILTER, lkAdapter, pyramidalLevels, kLevels, FILTER_OPT, lsAdapter, warping=False)

save_flow(U, V, os.path.join('.', 'denseLK_Fs2_0_PyrLvls2_LiuShen.mat'))
