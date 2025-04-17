# Create and activate the environment
conda env create -f environment.yml
conda activate optflow

# Verify installations
python -c "import numpy; import scipy; import skimage; import PIL; import pyopencl; import numba; import matplotlib"