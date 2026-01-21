"""Fuzzy logic inference system."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .membership_functions import FuzzySet


class FuzzyVariable:
    """
    Represents a fuzzy variable with multiple fuzzy sets.
    
    Attributes:
        name (str): Name of the variable
        range (Tuple[float, float]): Range of the variable
        fuzzy_sets (Dict[str, FuzzySet]): Dictionary of fuzzy sets
    """
    
    def __init__(self, name: str, var_range: Tuple[float, float]):
        """
        Initialize a fuzzy variable.
        
        Args:
            name: Name of the variable
            var_range: Range of the variable (min, max)
        """
        self.name = name
        self.range = var_range
        self.fuzzy_sets = {}
    
    def add_fuzzy_set(self, fuzzy_set: FuzzySet):
        """
        Add a fuzzy set to this variable.
        
        Args:
            fuzzy_set: FuzzySet to add
        """
        self.fuzzy_sets[fuzzy_set.name] = fuzzy_set
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp value.
        
        Args:
            value: Crisp input value
            
        Returns:
            Dictionary of membership degrees for each fuzzy set
        """
        result = {}
        for name, fuzzy_set in self.fuzzy_sets.items():
            result[name] = float(fuzzy_set.membership(np.array([value]))[0])
        return result


class FuzzyRule:
    """
    Represents a fuzzy IF-THEN rule.
    
    Attributes:
        antecedents (List[Tuple[str, str]]): List of (variable_name, fuzzy_set_name) for IF part
        consequent (Tuple[str, str]): (variable_name, fuzzy_set_name) for THEN part
        operator (str): 'and' or 'or' for combining antecedents
    """
    
    def __init__(
        self,
        antecedents: List[Tuple[str, str]],
        consequent: Tuple[str, str],
        operator: str = 'and'
    ):
        """
        Initialize a fuzzy rule.
        
        Args:
            antecedents: List of (variable_name, fuzzy_set_name) conditions
            consequent: (variable_name, fuzzy_set_name) result
            operator: 'and' or 'or' for combining antecedents
        """
        self.antecedents = antecedents
        self.consequent = consequent
        self.operator = operator
    
    def evaluate(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate the rule strength.
        
        Args:
            fuzzified_inputs: Dictionary of fuzzified input values
            
        Returns:
            Rule firing strength
        """
        strengths = []
        for var_name, fuzzy_set_name in self.antecedents:
            if var_name in fuzzified_inputs and fuzzy_set_name in fuzzified_inputs[var_name]:
                strengths.append(fuzzified_inputs[var_name][fuzzy_set_name])
        
        if not strengths:
            return 0.0
        
        if self.operator == 'and':
            return min(strengths)
        else:  # or
            return max(strengths)


class FuzzySystem:
    """
    Fuzzy Inference System (Mamdani type).
    
    Attributes:
        name (str): Name of the system
        input_variables (Dict[str, FuzzyVariable]): Input variables
        output_variables (Dict[str, FuzzyVariable]): Output variables
        rules (List[FuzzyRule]): Fuzzy rules
    """
    
    def __init__(self, name: str = "FuzzySystem"):
        """
        Initialize fuzzy system.
        
        Args:
            name: Name of the system
        """
        self.name = name
        self.input_variables = {}
        self.output_variables = {}
        self.rules = []
    
    def add_input_variable(self, variable: FuzzyVariable):
        """
        Add an input variable.
        
        Args:
            variable: FuzzyVariable to add
        """
        self.input_variables[variable.name] = variable
    
    def add_output_variable(self, variable: FuzzyVariable):
        """
        Add an output variable.
        
        Args:
            variable: FuzzyVariable to add
        """
        self.output_variables[variable.name] = variable
    
    def add_rule(self, rule: FuzzyRule):
        """
        Add a fuzzy rule.
        
        Args:
            rule: FuzzyRule to add
        """
        self.rules.append(rule)
    
    def inference(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Perform fuzzy inference.
        
        Args:
            inputs: Dictionary of crisp input values
            
        Returns:
            Dictionary of crisp output values
        """
        # Fuzzification
        fuzzified_inputs = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                fuzzified_inputs[var_name] = self.input_variables[var_name].fuzzify(value)
        
        # Rule evaluation and aggregation
        aggregated_outputs = {}
        for output_var_name in self.output_variables:
            aggregated_outputs[output_var_name] = {}
        
        for rule in self.rules:
            strength = rule.evaluate(fuzzified_inputs)
            output_var_name, output_fuzzy_set_name = rule.consequent
            
            if output_var_name not in aggregated_outputs:
                aggregated_outputs[output_var_name] = {}
            
            if output_fuzzy_set_name not in aggregated_outputs[output_var_name]:
                aggregated_outputs[output_var_name][output_fuzzy_set_name] = strength
            else:
                # Use max for aggregation
                aggregated_outputs[output_var_name][output_fuzzy_set_name] = max(
                    aggregated_outputs[output_var_name][output_fuzzy_set_name],
                    strength
                )
        
        # Defuzzification
        outputs = {}
        for var_name, fuzzy_values in aggregated_outputs.items():
            outputs[var_name] = self.defuzzify(var_name, fuzzy_values)
        
        return outputs
    
    def defuzzify(
        self,
        var_name: str,
        fuzzy_values: Dict[str, float],
        method: str = 'centroid'
    ) -> float:
        """
        Defuzzify fuzzy output to crisp value.
        
        Args:
            var_name: Name of the output variable
            fuzzy_values: Dictionary of fuzzy set activations
            method: Defuzzification method ('centroid', 'bisector', 'mom', 'som', 'lom')
            
        Returns:
            Crisp output value
        """
        if var_name not in self.output_variables:
            return 0.0
        
        output_var = self.output_variables[var_name]
        var_min, var_max = output_var.range
        
        # Create universe of discourse
        x = np.linspace(var_min, var_max, 1000)
        aggregated_mf = np.zeros_like(x)
        
        # Aggregate membership functions
        for fuzzy_set_name, strength in fuzzy_values.items():
            if fuzzy_set_name in output_var.fuzzy_sets:
                fuzzy_set = output_var.fuzzy_sets[fuzzy_set_name]
                mf_values = fuzzy_set.membership(x)
                # Apply strength using min (clipping)
                aggregated_mf = np.maximum(aggregated_mf, np.minimum(mf_values, strength))
        
        # Defuzzification
        if method == 'centroid':
            # Center of gravity
            if np.sum(aggregated_mf) == 0:
                return (var_min + var_max) / 2
            return np.sum(x * aggregated_mf) / np.sum(aggregated_mf)
        
        elif method == 'bisector':
            # Bisector of area
            total_area = np.sum(aggregated_mf)
            cumsum = np.cumsum(aggregated_mf)
            idx = np.where(cumsum >= total_area / 2)[0]
            return x[idx[0]] if len(idx) > 0 else (var_min + var_max) / 2
        
        elif method == 'mom':
            # Mean of maximum
            max_val = np.max(aggregated_mf)
            if max_val == 0:
                return (var_min + var_max) / 2
            max_indices = np.where(aggregated_mf == max_val)[0]
            return np.mean(x[max_indices])
        
        elif method == 'som':
            # Smallest of maximum
            max_val = np.max(aggregated_mf)
            if max_val == 0:
                return var_min
            max_indices = np.where(aggregated_mf == max_val)[0]
            return x[max_indices[0]]
        
        elif method == 'lom':
            # Largest of maximum
            max_val = np.max(aggregated_mf)
            if max_val == 0:
                return var_max
            max_indices = np.where(aggregated_mf == max_val)[0]
            return x[max_indices[-1]]
        
        else:
            # Default to centroid
            if np.sum(aggregated_mf) == 0:
                return (var_min + var_max) / 2
            return np.sum(x * aggregated_mf) / np.sum(aggregated_mf)
    
    def __repr__(self):
        return f"FuzzySystem(name={self.name}, inputs={len(self.input_variables)}, outputs={len(self.output_variables)}, rules={len(self.rules)})"
