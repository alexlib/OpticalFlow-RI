#!/usr/bin/env python
"""
Wrapper class for genericPyramidalOpticalFlow function to make it easier to use in benchmarks
"""

from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow

class GenericPyramidalOpticalFlowWrapper:
    """
    Wrapper class for genericPyramidalOpticalFlow function
    """
    def __init__(self, algo_adapter, filter_sigma=0.0, pyr_levels=1, k_levels=1, 
                 filter_opt=None, optional_algo_adapter=None, warping=True, 
                 bi_linear=True, pyramidal_intermediate_scaling=True, pyramidal_scaling=False):
        """
        Initialize the wrapper
        
        Args:
            algo_adapter: Adapter object for the main Optical Flow algorithm
            filter_sigma: Gaussian kernel filter size
            pyr_levels: Number of pyramidal levels
            k_levels: Number of iterations at each pyramidal level
            filter_opt: Optional filter sigma for the optional algorithm
            optional_algo_adapter: Optional adapter object for enhancement
            warping: Apply warping to the image
            bi_linear: Apply Bi-Linear interpolation for sub-pixel warping
            pyramidal_intermediate_scaling: Apply scaling when changing pyramidal level
            pyramidal_scaling: Apply scaling at the final pyramidal level
        """
        self.algo_adapter = algo_adapter
        self.filter_sigma = filter_sigma
        self.pyr_levels = pyr_levels
        self.k_levels = k_levels
        self.filter_opt = filter_opt
        self.optional_algo_adapter = optional_algo_adapter
        self.warping = warping
        self.bi_linear = bi_linear
        self.pyramidal_intermediate_scaling = pyramidal_intermediate_scaling
        self.pyramidal_scaling = pyramidal_scaling
    
    def calculateFlow(self, im1, im2):
        """
        Calculate optical flow between two images
        
        Args:
            im1: First image
            im2: Second image
            
        Returns:
            U, V: Horizontal and vertical velocity components
        """
        return genericPyramidalOpticalFlow(
            im1, im2, 
            self.filter_sigma, 
            self.algo_adapter, 
            pyramidalLevels=self.pyr_levels,
            kLevels=self.k_levels,
            FILTER_OPT=self.filter_opt,
            optionalOFlowAlgoAdapter=self.optional_algo_adapter,
            warping=self.warping,
            biLinear=self.bi_linear,
            pyramidalIntermediateScaling=self.pyramidal_intermediate_scaling,
            pyramidalScaling=self.pyramidal_scaling
        )
