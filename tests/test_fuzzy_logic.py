"""Unit tests for Fuzzy Logic module."""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fuzzy_logic.membership_functions import MembershipFunction, FuzzySet
from src.fuzzy_logic.fuzzy_system import FuzzyVariable, FuzzyRule, FuzzySystem


class TestMembershipFunctions(unittest.TestCase):
    """Test cases for membership functions."""
    
    def test_triangular(self):
        """Test triangular membership function."""
        x = np.array([0, 0.5, 1, 1.5, 2])
        mf = MembershipFunction.triangular(x, 0, 1, 2)
        
        self.assertAlmostEqual(mf[0], 0)
        self.assertAlmostEqual(mf[1], 0.5)
        self.assertAlmostEqual(mf[2], 1)
        self.assertAlmostEqual(mf[3], 0.5)
        self.assertAlmostEqual(mf[4], 0)
    
    def test_trapezoidal(self):
        """Test trapezoidal membership function."""
        x = np.array([0, 0.5, 1, 2, 2.5, 3])
        mf = MembershipFunction.trapezoidal(x, 0, 1, 2, 3)
        
        self.assertAlmostEqual(mf[0], 0)
        self.assertAlmostEqual(mf[1], 0.5)
        self.assertAlmostEqual(mf[2], 1)
        self.assertAlmostEqual(mf[3], 1)
        self.assertAlmostEqual(mf[4], 0.5)
        self.assertAlmostEqual(mf[5], 0)
    
    def test_gaussian(self):
        """Test Gaussian membership function."""
        x = np.array([0, 0.5, 1])
        mf = MembershipFunction.gaussian(x, 0.5, 0.5)
        
        self.assertTrue(mf[1] > mf[0])
        self.assertTrue(mf[1] > mf[2])
        self.assertAlmostEqual(mf[1], 1.0, places=5)


class TestFuzzySystem(unittest.TestCase):
    """Test cases for Fuzzy System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fuzzy_system = FuzzySystem("Test System")
        
        # Create input variable
        temp = FuzzyVariable("temperature", (0, 100))
        temp.add_fuzzy_set(FuzzySet("cold", "triangular", (0, 0, 50)))
        temp.add_fuzzy_set(FuzzySet("warm", "triangular", (25, 50, 75)))
        temp.add_fuzzy_set(FuzzySet("hot", "triangular", (50, 100, 100)))
        
        # Create output variable
        fan = FuzzyVariable("fan_speed", (0, 100))
        fan.add_fuzzy_set(FuzzySet("slow", "triangular", (0, 0, 50)))
        fan.add_fuzzy_set(FuzzySet("medium", "triangular", (25, 50, 75)))
        fan.add_fuzzy_set(FuzzySet("fast", "triangular", (50, 100, 100)))
        
        self.fuzzy_system.add_input_variable(temp)
        self.fuzzy_system.add_output_variable(fan)
        
        # Add rules
        self.fuzzy_system.add_rule(FuzzyRule([("temperature", "cold")], ("fan_speed", "slow")))
        self.fuzzy_system.add_rule(FuzzyRule([("temperature", "warm")], ("fan_speed", "medium")))
        self.fuzzy_system.add_rule(FuzzyRule([("temperature", "hot")], ("fan_speed", "fast")))
    
    def test_fuzzification(self):
        """Test fuzzification."""
        var = self.fuzzy_system.input_variables["temperature"]
        result = var.fuzzify(25)
        
        self.assertIn("cold", result)
        self.assertIn("warm", result)
        self.assertIn("hot", result)
        self.assertTrue(all(0 <= v <= 1 for v in result.values()))
    
    def test_inference(self):
        """Test fuzzy inference."""
        output = self.fuzzy_system.inference({"temperature": 75})
        
        self.assertIn("fan_speed", output)
        self.assertTrue(0 <= output["fan_speed"] <= 100)
    
    def test_defuzzification(self):
        """Test different defuzzification methods."""
        fuzzy_values = {"slow": 0.3, "medium": 0.7, "fast": 0.0}
        
        centroid = self.fuzzy_system.defuzzify("fan_speed", fuzzy_values, method="centroid")
        mom = self.fuzzy_system.defuzzify("fan_speed", fuzzy_values, method="mom")
        
        self.assertTrue(0 <= centroid <= 100)
        self.assertTrue(0 <= mom <= 100)


if __name__ == '__main__':
    unittest.main()
