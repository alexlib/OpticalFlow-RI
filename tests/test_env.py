import numpy
import scipy
import skimage
import PIL
import numba
import matplotlib

print("NumPy version:", numpy.__version__)
print("SciPy version:", scipy.__version__)
print("scikit-image version:", skimage.__version__)
print("PIL version:", PIL.__version__)
print("Numba version:", numba.__version__)
print("Matplotlib version:", matplotlib.__version__)

try:
    import pyopencl
    print("PyOpenCL version:", pyopencl.__version__)
    print("All required packages are installed and working correctly!")
except ImportError as e:
    print("Error importing PyOpenCL:", e)
    print("All other packages are installed and working correctly!")
