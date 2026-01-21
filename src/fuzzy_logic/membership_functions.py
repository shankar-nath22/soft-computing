"""Membership functions for fuzzy logic systems."""

import numpy as np
from typing import Callable, Tuple


class MembershipFunction:
    """
    Fuzzy membership functions.
    
    Provides various membership function types for fuzzy sets.
    """
    
    @staticmethod
    def triangular(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Triangular membership function.
        
        Args:
            x: Input values
            a: Left foot
            b: Peak
            c: Right foot
            
        Returns:
            Membership degrees
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        
        # Left slope
        mask1 = (x >= a) & (x <= b)
        result[mask1] = (x[mask1] - a) / (b - a) if b != a else 0
        
        # Right slope
        mask2 = (x > b) & (x <= c)
        result[mask2] = (c - x[mask2]) / (c - b) if c != b else 0
        
        return result
    
    @staticmethod
    def trapezoidal(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """
        Trapezoidal membership function.
        
        Args:
            x: Input values
            a: Left foot
            b: Left shoulder
            c: Right shoulder
            d: Right foot
            
        Returns:
            Membership degrees
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        
        # Left slope
        mask1 = (x >= a) & (x < b)
        result[mask1] = (x[mask1] - a) / (b - a) if b != a else 0
        
        # Flat top
        mask2 = (x >= b) & (x <= c)
        result[mask2] = 1.0
        
        # Right slope
        mask3 = (x > c) & (x <= d)
        result[mask3] = (d - x[mask3]) / (d - c) if d != c else 0
        
        return result
    
    @staticmethod
    def gaussian(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
        """
        Gaussian membership function.
        
        Args:
            x: Input values
            mean: Mean of the Gaussian
            sigma: Standard deviation
            
        Returns:
            Membership degrees
        """
        x = np.asarray(x)
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    @staticmethod
    def bell(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Generalized bell membership function.
        
        Args:
            x: Input values
            a: Width parameter
            b: Slope parameter
            c: Center
            
        Returns:
            Membership degrees
        """
        x = np.asarray(x)
        return 1 / (1 + np.abs((x - c) / a) ** (2 * b))
    
    @staticmethod
    def sigmoid(x: np.ndarray, a: float, c: float) -> np.ndarray:
        """
        Sigmoid membership function.
        
        Args:
            x: Input values
            a: Slope
            c: Crossover point
            
        Returns:
            Membership degrees
        """
        x = np.asarray(x)
        return 1 / (1 + np.exp(-a * (x - c)))


class FuzzySet:
    """
    Represents a fuzzy set with a membership function.
    
    Attributes:
        name (str): Name of the fuzzy set
        mf_type (str): Type of membership function
        params (Tuple): Parameters for the membership function
    """
    
    def __init__(
        self,
        name: str,
        mf_type: str = 'triangular',
        params: Tuple = (0, 0.5, 1)
    ):
        """
        Initialize a fuzzy set.
        
        Args:
            name: Name of the fuzzy set
            mf_type: Type of membership function ('triangular', 'trapezoidal', 'gaussian', 'bell', 'sigmoid')
            params: Parameters for the membership function
        """
        self.name = name
        self.mf_type = mf_type
        self.params = params
        self.mf = self._get_membership_function()
    
    def _get_membership_function(self) -> Callable:
        """Get the membership function based on type."""
        mf_functions = {
            'triangular': lambda x: MembershipFunction.triangular(x, *self.params),
            'trapezoidal': lambda x: MembershipFunction.trapezoidal(x, *self.params),
            'gaussian': lambda x: MembershipFunction.gaussian(x, *self.params),
            'bell': lambda x: MembershipFunction.bell(x, *self.params),
            'sigmoid': lambda x: MembershipFunction.sigmoid(x, *self.params),
        }
        return mf_functions.get(self.mf_type, mf_functions['triangular'])
    
    def membership(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate membership degree for input values.
        
        Args:
            x: Input values
            
        Returns:
            Membership degrees
        """
        return self.mf(x)
    
    def __repr__(self):
        return f"FuzzySet(name={self.name}, type={self.mf_type}, params={self.params})"
