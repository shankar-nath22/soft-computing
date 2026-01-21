"""Core genetic algorithm implementation."""

import numpy as np
from typing import Callable, Optional, Tuple, List
from .operators import Selection, Crossover, Mutation


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimization.
    
    Attributes:
        population_size (int): Size of the population
        chromosome_length (int): Length of each chromosome
        generations (int): Number of generations
        crossover_rate (float): Probability of crossover
        mutation_rate (float): Probability of mutation
        elitism_count (int): Number of elite individuals to preserve
        fitness_function (Callable): Function to evaluate fitness
    """
    
    def __init__(
        self,
        population_size: int = 100,
        chromosome_length: int = 10,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        elitism_count: int = 2,
        fitness_function: Optional[Callable] = None,
        bounds: Tuple[float, float] = (0, 1),
        encoding: str = 'real'
    ):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in population
            chromosome_length: Number of genes in chromosome
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
            fitness_function: Function to evaluate fitness
            bounds: (min, max) bounds for gene values (real encoding)
            encoding: Encoding type ('real' or 'binary')
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.encoding = encoding
        
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': []
        }
    
    def initialize_population(self):
        """Initialize random population."""
        if self.encoding == 'binary':
            self.population = np.random.randint(
                0, 2, (self.population_size, self.chromosome_length)
            ).astype(float)
        else:  # real encoding
            self.population = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                (self.population_size, self.chromosome_length)
            )
    
    def evaluate_fitness(self):
        """Evaluate fitness of all individuals in population."""
        if self.fitness_function is None:
            raise ValueError("Fitness function not defined")
        
        self.fitness = np.array([
            self.fitness_function(individual)
            for individual in self.population
        ])
    
    def select_parents(
        self,
        method: str = 'tournament',
        tournament_size: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select two parents for reproduction.
        
        Args:
            method: Selection method ('tournament', 'roulette', 'rank')
            tournament_size: Size of tournament (for tournament selection)
            
        Returns:
            Tuple of two parent individuals
        """
        if method == 'tournament':
            parent1 = Selection.tournament(self.population, self.fitness, tournament_size)
            parent2 = Selection.tournament(self.population, self.fitness, tournament_size)
        elif method == 'roulette':
            parent1 = Selection.roulette_wheel(self.population, self.fitness)
            parent2 = Selection.roulette_wheel(self.population, self.fitness)
        elif method == 'rank':
            parent1 = Selection.rank_based(self.population, self.fitness)
            parent2 = Selection.rank_based(self.population, self.fitness)
        else:
            parent1 = Selection.tournament(self.population, self.fitness, tournament_size)
            parent2 = Selection.tournament(self.population, self.fitness, tournament_size)
        
        return parent1, parent2
    
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        method: str = 'single_point'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            method: Crossover method ('single_point', 'two_point', 'uniform', 'arithmetic')
            
        Returns:
            Tuple of two offspring
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if method == 'single_point':
            return Crossover.single_point(parent1, parent2)
        elif method == 'two_point':
            return Crossover.two_point(parent1, parent2)
        elif method == 'uniform':
            return Crossover.uniform(parent1, parent2)
        elif method == 'arithmetic':
            return Crossover.arithmetic(parent1, parent2)
        else:
            return Crossover.single_point(parent1, parent2)
    
    def mutate(
        self,
        individual: np.ndarray,
        method: str = 'gaussian'
    ) -> np.ndarray:
        """
        Perform mutation.
        
        Args:
            individual: Individual to mutate
            method: Mutation method ('gaussian', 'uniform', 'bit_flip', 'swap', 'polynomial')
            
        Returns:
            Mutated individual
        """
        if self.encoding == 'binary':
            return Mutation.bit_flip(individual, self.mutation_rate)
        
        if method == 'gaussian':
            return Mutation.gaussian(individual, self.mutation_rate)
        elif method == 'uniform':
            return Mutation.uniform(individual, self.mutation_rate, self.bounds)
        elif method == 'swap':
            return Mutation.swap(individual, self.mutation_rate)
        elif method == 'polynomial':
            return Mutation.polynomial(individual, self.mutation_rate, bounds=self.bounds)
        else:
            return Mutation.gaussian(individual, self.mutation_rate)
    
    def evolve(
        self,
        selection_method: str = 'tournament',
        crossover_method: str = 'single_point',
        mutation_method: str = 'gaussian',
        verbose: bool = True
    ):
        """
        Run the genetic algorithm.
        
        Args:
            selection_method: Method for parent selection
            crossover_method: Method for crossover
            mutation_method: Method for mutation
            verbose: Whether to print progress
        """
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            self.evaluate_fitness()
            
            # Track best individual
            best_idx = np.argmax(self.fitness)
            if self.best_fitness is None or self.fitness[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            # Record history
            self.history['best_fitness'].append(np.max(self.fitness))
            self.history['avg_fitness'].append(np.mean(self.fitness))
            self.history['worst_fitness'].append(np.min(self.fitness))
            
            if verbose and (generation + 1) % max(1, self.generations // 10) == 0:
                print(f"Generation {generation + 1}/{self.generations}, "
                      f"Best: {np.max(self.fitness):.4f}, "
                      f"Avg: {np.mean(self.fitness):.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            if self.elitism_count > 0:
                elite_indices = np.argsort(self.fitness)[-self.elitism_count:]
                for idx in elite_indices:
                    new_population.append(self.population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.select_parents(selection_method)
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2, crossover_method)
                
                # Mutation
                offspring1 = self.mutate(offspring1, mutation_method)
                offspring2 = self.mutate(offspring2, mutation_method)
                
                # Ensure bounds for real encoding
                if self.encoding == 'real':
                    offspring1 = np.clip(offspring1, self.bounds[0], self.bounds[1])
                    offspring2 = np.clip(offspring2, self.bounds[0], self.bounds[1])
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Update population
            self.population = np.array(new_population[:self.population_size])
        
        # Final evaluation
        self.evaluate_fitness()
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_individual = self.population[best_idx].copy()
    
    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        """
        Get the best solution found.
        
        Returns:
            Tuple of (best_individual, best_fitness)
        """
        return self.best_individual, self.best_fitness
