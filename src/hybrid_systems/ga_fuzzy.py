"""GA-optimized Fuzzy System for control applications."""

import numpy as np
from typing import Callable, Optional, Tuple
import sys
sys.path.append('..')
from src.genetic_algorithm.ga_core import GeneticAlgorithm
from src.fuzzy_logic.fuzzy_system import FuzzySystem, FuzzyVariable, FuzzyRule
from src.fuzzy_logic.membership_functions import FuzzySet


class GAFuzzy:
    """
    Genetic Algorithm optimized Fuzzy Logic System.
    
    Uses GA to optimize fuzzy membership function parameters and rule weights
    for control or classification tasks.
    
    Attributes:
        fuzzy_system (FuzzySystem): The fuzzy inference system
        ga (GeneticAlgorithm): Genetic algorithm optimizer
        control_function (Callable): Function to evaluate control performance
    """
    
    def __init__(
        self,
        input_ranges: dict,
        output_ranges: dict,
        n_mf_per_variable: int = 3,
        population_size: int = 50,
        generations: int = 100
    ):
        """
        Initialize GA-Fuzzy system.
        
        Args:
            input_ranges: Dict of {var_name: (min, max)} for inputs
            output_ranges: Dict of {var_name: (min, max)} for outputs
            n_mf_per_variable: Number of membership functions per variable
            population_size: Population size for GA
            generations: Number of GA generations
        """
        self.input_ranges = input_ranges
        self.output_ranges = output_ranges
        self.n_mf = n_mf_per_variable
        self.population_size = population_size
        self.generations = generations
        
        self.fuzzy_system = None
        self.ga = None
        self.control_function = None
        self.best_params = None
        
        # Calculate chromosome length
        # For each variable: n_mf triangular MF parameters (a, b, c)
        n_inputs = len(input_ranges)
        n_outputs = len(output_ranges)
        self.chromosome_length = (n_inputs + n_outputs) * n_mf_per_variable * 3
    
    def _decode_chromosome(self, chromosome: np.ndarray) -> dict:
        """
        Decode chromosome into fuzzy system parameters.
        
        Args:
            chromosome: GA chromosome
            
        Returns:
            Dictionary of parameters for each variable
        """
        params = {}
        idx = 0
        
        # Decode input variables
        for var_name, (var_min, var_max) in self.input_ranges.items():
            mf_params = []
            for _ in range(self.n_mf):
                # Each MF has 3 parameters (a, b, c) for triangular function
                a = var_min + chromosome[idx] * (var_max - var_min)
                b = var_min + chromosome[idx + 1] * (var_max - var_min)
                c = var_min + chromosome[idx + 2] * (var_max - var_min)
                
                # Ensure proper ordering: a <= b <= c
                params_sorted = sorted([a, b, c])
                mf_params.append(tuple(params_sorted))
                idx += 3
            params[var_name] = mf_params
        
        # Decode output variables
        for var_name, (var_min, var_max) in self.output_ranges.items():
            mf_params = []
            for _ in range(self.n_mf):
                a = var_min + chromosome[idx] * (var_max - var_min)
                b = var_min + chromosome[idx + 1] * (var_max - var_min)
                c = var_min + chromosome[idx + 2] * (var_max - var_min)
                
                params_sorted = sorted([a, b, c])
                mf_params.append(tuple(params_sorted))
                idx += 3
            params[var_name] = mf_params
        
        return params
    
    def _build_fuzzy_system(self, params: dict) -> FuzzySystem:
        """
        Build fuzzy system from parameters.
        
        Args:
            params: Dictionary of MF parameters
            
        Returns:
            Configured FuzzySystem
        """
        fs = FuzzySystem("GA-Optimized Fuzzy System")
        
        # Add input variables
        for var_name, (var_min, var_max) in self.input_ranges.items():
            var = FuzzyVariable(var_name, (var_min, var_max))
            
            mf_names = ['low', 'medium', 'high'] if self.n_mf == 3 else [f'mf{i}' for i in range(self.n_mf)]
            for i, mf_name in enumerate(mf_names[:self.n_mf]):
                fuzzy_set = FuzzySet(
                    name=mf_name,
                    mf_type='triangular',
                    params=params[var_name][i]
                )
                var.add_fuzzy_set(fuzzy_set)
            
            fs.add_input_variable(var)
        
        # Add output variables
        for var_name, (var_min, var_max) in self.output_ranges.items():
            var = FuzzyVariable(var_name, (var_min, var_max))
            
            mf_names = ['low', 'medium', 'high'] if self.n_mf == 3 else [f'mf{i}' for i in range(self.n_mf)]
            for i, mf_name in enumerate(mf_names[:self.n_mf]):
                fuzzy_set = FuzzySet(
                    name=mf_name,
                    mf_type='triangular',
                    params=params[var_name][i]
                )
                var.add_fuzzy_set(fuzzy_set)
            
            fs.add_output_variable(var)
        
        # Add default rules (can be customized)
        self._add_default_rules(fs)
        
        return fs
    
    def _add_default_rules(self, fs: FuzzySystem):
        """
        Add default fuzzy rules to the system.
        
        Args:
            fs: Fuzzy system
        """
        input_names = list(self.input_ranges.keys())
        output_names = list(self.output_ranges.keys())
        
        mf_names = ['low', 'medium', 'high'] if self.n_mf == 3 else [f'mf{i}' for i in range(self.n_mf)]
        
        # Create simple rules based on number of inputs
        if len(input_names) == 1:
            for mf in mf_names[:self.n_mf]:
                rule = FuzzyRule(
                    antecedents=[(input_names[0], mf)],
                    consequent=(output_names[0], mf)
                )
                fs.add_rule(rule)
        
        elif len(input_names) == 2:
            # Create rules for two inputs
            for mf1 in mf_names[:self.n_mf]:
                for mf2 in mf_names[:self.n_mf]:
                    # Simple mapping: average of inputs
                    mf1_idx = mf_names.index(mf1)
                    mf2_idx = mf_names.index(mf2)
                    avg_idx = min((mf1_idx + mf2_idx) // 2, self.n_mf - 1)
                    output_mf = mf_names[avg_idx]
                    
                    rule = FuzzyRule(
                        antecedents=[(input_names[0], mf1), (input_names[1], mf2)],
                        consequent=(output_names[0], output_mf)
                    )
                    fs.add_rule(rule)
    
    def _fitness_function(self, chromosome: np.ndarray) -> float:
        """
        Fitness function for GA.
        
        Args:
            chromosome: GA chromosome
            
        Returns:
            Fitness score
        """
        # Decode chromosome
        params = self._decode_chromosome(chromosome)
        
        # Build fuzzy system
        fs = self._build_fuzzy_system(params)
        
        # Evaluate using control function
        if self.control_function is None:
            return 0.0
        
        try:
            fitness = self.control_function(fs)
            return fitness if not np.isnan(fitness) else 0.0
        except:
            return 0.0
    
    def optimize(
        self,
        control_function: Callable,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        verbose: bool = True
    ):
        """
        Optimize fuzzy system parameters using GA.
        
        Args:
            control_function: Function that evaluates fuzzy system performance
            crossover_rate: GA crossover rate
            mutation_rate: GA mutation rate
            verbose: Whether to print progress
        """
        self.control_function = control_function
        
        # Initialize GA
        self.ga = GeneticAlgorithm(
            population_size=self.population_size,
            chromosome_length=self.chromosome_length,
            generations=self.generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_count=2,
            fitness_function=self._fitness_function,
            bounds=(0, 1),
            encoding='real'
        )
        
        # Run optimization
        self.ga.evolve(
            selection_method='tournament',
            crossover_method='arithmetic',
            mutation_method='gaussian',
            verbose=verbose
        )
        
        # Get best solution
        self.best_params, best_fitness = self.ga.get_best_solution()
        
        # Build final fuzzy system
        params = self._decode_chromosome(self.best_params)
        self.fuzzy_system = self._build_fuzzy_system(params)
        
        if verbose:
            print(f"\nOptimization complete. Best fitness: {best_fitness:.4f}")
    
    def predict(self, inputs: dict) -> dict:
        """
        Make prediction using optimized fuzzy system.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values
        """
        if self.fuzzy_system is None:
            raise ValueError("System not optimized yet. Call optimize() first.")
        
        return self.fuzzy_system.inference(inputs)
    
    def get_optimization_history(self) -> dict:
        """
        Get optimization history from GA.
        
        Returns:
            Dictionary with fitness evolution
        """
        if self.ga is None:
            return {}
        return self.ga.history
