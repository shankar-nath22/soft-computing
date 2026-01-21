"""Unit tests for Genetic Algorithm module."""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genetic_algorithm.ga_core import GeneticAlgorithm
from src.genetic_algorithm.operators import Selection, Crossover, Mutation


class TestOperators(unittest.TestCase):
    """Test cases for GA operators."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.population = np.random.rand(10, 5)
        self.fitness = np.random.rand(10)
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        selected = Selection.tournament(self.population, self.fitness, tournament_size=3)
        self.assertEqual(selected.shape, (5,))
    
    def test_roulette_wheel_selection(self):
        """Test roulette wheel selection."""
        selected = Selection.roulette_wheel(self.population, self.fitness)
        self.assertEqual(selected.shape, (5,))
    
    def test_rank_based_selection(self):
        """Test rank-based selection."""
        selected = Selection.rank_based(self.population, self.fitness)
        self.assertEqual(selected.shape, (5,))
    
    def test_single_point_crossover(self):
        """Test single-point crossover."""
        parent1 = np.array([1, 2, 3, 4, 5])
        parent2 = np.array([6, 7, 8, 9, 10])
        
        offspring1, offspring2 = Crossover.single_point(parent1, parent2)
        
        self.assertEqual(len(offspring1), 5)
        self.assertEqual(len(offspring2), 5)
    
    def test_uniform_crossover(self):
        """Test uniform crossover."""
        parent1 = np.array([1, 2, 3, 4, 5])
        parent2 = np.array([6, 7, 8, 9, 10])
        
        offspring1, offspring2 = Crossover.uniform(parent1, parent2)
        
        self.assertEqual(len(offspring1), 5)
        self.assertEqual(len(offspring2), 5)
    
    def test_gaussian_mutation(self):
        """Test Gaussian mutation."""
        individual = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        mutated = Mutation.gaussian(individual, mutation_rate=1.0, sigma=0.1)
        
        self.assertEqual(len(mutated), 5)
        self.assertFalse(np.array_equal(individual, mutated))


class TestGeneticAlgorithm(unittest.TestCase):
    """Test cases for Genetic Algorithm."""
    
    def test_initialization(self):
        """Test GA initialization."""
        def fitness_fn(x):
            return np.sum(x)
        
        ga = GeneticAlgorithm(
            population_size=20,
            chromosome_length=5,
            generations=10,
            fitness_function=fitness_fn
        )
        
        self.assertEqual(ga.population_size, 20)
        self.assertEqual(ga.chromosome_length, 5)
        self.assertEqual(ga.generations, 10)
    
    def test_population_initialization(self):
        """Test population initialization."""
        def fitness_fn(x):
            return np.sum(x)
        
        ga = GeneticAlgorithm(
            population_size=20,
            chromosome_length=5,
            generations=10,
            fitness_function=fitness_fn,
            encoding='real'
        )
        ga.initialize_population()
        
        self.assertEqual(ga.population.shape, (20, 5))
    
    def test_evolution(self):
        """Test evolution process."""
        def fitness_fn(x):
            # Maximize sum of genes
            return np.sum(x)
        
        ga = GeneticAlgorithm(
            population_size=20,
            chromosome_length=5,
            generations=10,
            fitness_function=fitness_fn,
            encoding='real',
            bounds=(0, 1)
        )
        
        ga.evolve(verbose=False)
        
        self.assertIsNotNone(ga.best_individual)
        self.assertIsNotNone(ga.best_fitness)
        self.assertEqual(len(ga.history['best_fitness']), 10)
        
        # Check if fitness improved
        self.assertGreater(ga.history['best_fitness'][-1], ga.history['best_fitness'][0])
    
    def test_binary_encoding(self):
        """Test binary encoding."""
        def fitness_fn(x):
            return np.sum(x)
        
        ga = GeneticAlgorithm(
            population_size=20,
            chromosome_length=10,
            generations=5,
            fitness_function=fitness_fn,
            encoding='binary'
        )
        ga.evolve(verbose=False)
        
        self.assertTrue(np.all((ga.population == 0) | (ga.population == 1)))


if __name__ == '__main__':
    unittest.main()
