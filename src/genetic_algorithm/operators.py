"""Genetic algorithm operators: selection, crossover, and mutation."""

import numpy as np
from typing import Tuple, List


class Selection:
    """Selection operators for genetic algorithms."""
    
    @staticmethod
    def tournament(
        population: np.ndarray,
        fitness: np.ndarray,
        tournament_size: int = 3
    ) -> np.ndarray:
        """
        Tournament selection.
        
        Args:
            population: Population array
            fitness: Fitness values
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual
        """
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    @staticmethod
    def roulette_wheel(
        population: np.ndarray,
        fitness: np.ndarray
    ) -> np.ndarray:
        """
        Roulette wheel selection.
        
        Args:
            population: Population array
            fitness: Fitness values
            
        Returns:
            Selected individual
        """
        # Ensure non-negative fitness
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            adjusted_fitness = fitness - min_fitness + 1e-10
        else:
            adjusted_fitness = fitness + 1e-10
        
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        idx = np.random.choice(len(population), p=probabilities)
        return population[idx].copy()
    
    @staticmethod
    def rank_based(
        population: np.ndarray,
        fitness: np.ndarray
    ) -> np.ndarray:
        """
        Rank-based selection.
        
        Args:
            population: Population array
            fitness: Fitness values
            
        Returns:
            Selected individual
        """
        ranks = np.argsort(np.argsort(fitness)) + 1
        probabilities = ranks / np.sum(ranks)
        idx = np.random.choice(len(population), p=probabilities)
        return population[idx].copy()


class Crossover:
    """Crossover operators for genetic algorithms."""
    
    @staticmethod
    def single_point(
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, len(parent1))
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        return offspring1, offspring2
    
    @staticmethod
    def two_point(
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        if len(parent1) <= 2:
            return Crossover.single_point(parent1, parent2)
        
        points = sorted(np.random.choice(range(1, len(parent1)), 2, replace=False))
        point1, point2 = points
        
        offspring1 = np.concatenate([
            parent1[:point1],
            parent2[point1:point2],
            parent1[point2:]
        ])
        offspring2 = np.concatenate([
            parent2[:point1],
            parent1[point1:point2],
            parent2[point2:]
        ])
        return offspring1, offspring2
    
    @staticmethod
    def uniform(
        parent1: np.ndarray,
        parent2: np.ndarray,
        prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            prob: Probability of selecting from parent1
            
        Returns:
            Tuple of two offspring
        """
        mask = np.random.random(len(parent1)) < prob
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        return offspring1, offspring2
    
    @staticmethod
    def arithmetic(
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arithmetic crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            alpha: Blending factor
            
        Returns:
            Tuple of two offspring
        """
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = (1 - alpha) * parent1 + alpha * parent2
        return offspring1, offspring2


class Mutation:
    """Mutation operators for genetic algorithms."""
    
    @staticmethod
    def bit_flip(
        individual: np.ndarray,
        mutation_rate: float = 0.01
    ) -> np.ndarray:
        """
        Bit flip mutation for binary encoding.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of flipping each bit
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        mask = np.random.random(len(individual)) < mutation_rate
        mutated[mask] = 1 - mutated[mask]
        return mutated
    
    @staticmethod
    def gaussian(
        individual: np.ndarray,
        mutation_rate: float = 0.1,
        sigma: float = 0.1
    ) -> np.ndarray:
        """
        Gaussian mutation for real-valued encoding.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutating each gene
            sigma: Standard deviation of Gaussian noise
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        mask = np.random.random(len(individual)) < mutation_rate
        mutated[mask] += np.random.normal(0, sigma, np.sum(mask))
        return mutated
    
    @staticmethod
    def uniform(
        individual: np.ndarray,
        mutation_rate: float = 0.1,
        bounds: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Uniform mutation for real-valued encoding.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutating each gene
            bounds: (min, max) bounds for gene values
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        mask = np.random.random(len(individual)) < mutation_rate
        mutated[mask] = np.random.uniform(bounds[0], bounds[1], np.sum(mask))
        return mutated
    
    @staticmethod
    def swap(
        individual: np.ndarray,
        mutation_rate: float = 0.1
    ) -> np.ndarray:
        """
        Swap mutation (swap two random positions).
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of performing swap
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        if np.random.random() < mutation_rate and len(individual) > 1:
            idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    @staticmethod
    def polynomial(
        individual: np.ndarray,
        mutation_rate: float = 0.1,
        eta: float = 20.0,
        bounds: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Polynomial mutation.
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutating each gene
            eta: Distribution index
            bounds: (min, max) bounds for gene values
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        mask = np.random.random(len(individual)) < mutation_rate
        
        for i in np.where(mask)[0]:
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            
            mutated[i] = mutated[i] + delta * (bounds[1] - bounds[0])
            mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])
        
        return mutated
