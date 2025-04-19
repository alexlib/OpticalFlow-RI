"""
Fix the Liu-Shen results to use the correct variable names.
"""

import os
import sys
import numpy as np
import scipy.io as sio

def fix_liushen_results():
    """Fix the Liu-Shen results to use the correct variable names."""
    # Load the MAT file
    mat_path = os.path.join(os.path.dirname(__file__), 'flow_results', 'liushen_parabolic', 'flow_results.mat')
    if not os.path.exists(mat_path):
        print(f"MAT file not found: {mat_path}")
        return
    
    # Load the data
    data = sio.loadmat(mat_path)
    
    # Create new data with the correct variable names
    new_data = {
        'U': data['u'],
        'V': data['v'],
        'error': data['error']
    }
    
    # Save the new data
    new_mat_path = os.path.join(os.path.dirname(__file__), 'flow_results', 'liushen_parabolic', 'flow_results_fixed.mat')
    sio.savemat(new_mat_path, new_data)
    
    print(f"Fixed MAT file saved to {new_mat_path}")

if __name__ == "__main__":
    fix_liushen_results()
