"""Control system example using GA-optimized fuzzy logic controller."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fuzzy_logic.fuzzy_system import FuzzySystem, FuzzyVariable, FuzzyRule
from src.fuzzy_logic.membership_functions import FuzzySet
from src.hybrid_systems.ga_fuzzy import GAFuzzy
from src.utils.visualization import Visualizer


class TemperatureController:
    """Simple temperature control simulation."""
    
    def __init__(self, target_temp: float = 25.0):
        """
        Initialize controller.
        
        Args:
            target_temp: Target temperature in Celsius
        """
        self.target_temp = target_temp
        self.current_temp = 20.0
        self.ambient_temp = 15.0
    
    def update(self, heater_power: float, dt: float = 0.1) -> float:
        """
        Update temperature based on heater power.
        
        Args:
            heater_power: Heater power (0-100%)
            dt: Time step
            
        Returns:
            New temperature
        """
        # Simple physics model
        heating_effect = heater_power * 0.5 * dt
        cooling_effect = (self.current_temp - self.ambient_temp) * 0.1 * dt
        
        self.current_temp += heating_effect - cooling_effect
        return self.current_temp
    
    def get_error(self) -> float:
        """Get temperature error."""
        return self.target_temp - self.current_temp


def create_fuzzy_controller() -> FuzzySystem:
    """Create a manual fuzzy temperature controller."""
    fs = FuzzySystem("Temperature Controller")
    
    # Input: Temperature Error
    error_var = FuzzyVariable("error", (-20, 20))
    error_var.add_fuzzy_set(FuzzySet("negative", "triangular", (-20, -20, 0)))
    error_var.add_fuzzy_set(FuzzySet("zero", "triangular", (-5, 0, 5)))
    error_var.add_fuzzy_set(FuzzySet("positive", "triangular", (0, 20, 20)))
    fs.add_input_variable(error_var)
    
    # Output: Heater Power
    power_var = FuzzyVariable("power", (0, 100))
    power_var.add_fuzzy_set(FuzzySet("low", "triangular", (0, 0, 50)))
    power_var.add_fuzzy_set(FuzzySet("medium", "triangular", (25, 50, 75)))
    power_var.add_fuzzy_set(FuzzySet("high", "triangular", (50, 100, 100)))
    fs.add_output_variable(power_var)
    
    # Rules
    fs.add_rule(FuzzyRule([("error", "negative")], ("power", "low")))
    fs.add_rule(FuzzyRule([("error", "zero")], ("power", "medium")))
    fs.add_rule(FuzzyRule([("error", "positive")], ("power", "high")))
    
    return fs


def simulate_control(fuzzy_controller: FuzzySystem, steps: int = 200) -> tuple:
    """
    Simulate temperature control.
    
    Args:
        fuzzy_controller: Fuzzy controller
        steps: Number of simulation steps
        
    Returns:
        Tuple of (time, temperature, setpoint, control_signal)
    """
    controller = TemperatureController(target_temp=25.0)
    
    time = []
    temperatures = []
    setpoints = []
    control_signals = []
    
    for i in range(steps):
        # Get current state
        error = controller.get_error()
        
        # Fuzzy inference
        outputs = fuzzy_controller.inference({"error": error})
        power = outputs.get("power", 50)
        
        # Update system
        temp = controller.update(power)
        
        # Record
        time.append(i * 0.1)
        temperatures.append(temp)
        setpoints.append(controller.target_temp)
        control_signals.append(power)
    
    return np.array(time), np.array(temperatures), np.array(setpoints), np.array(control_signals)


def evaluate_controller(fuzzy_controller: FuzzySystem) -> float:
    """
    Evaluate controller performance.
    
    Args:
        fuzzy_controller: Fuzzy controller
        
    Returns:
        Fitness score (lower is better for error)
    """
    time, temps, setpoints, _ = simulate_control(fuzzy_controller, steps=200)
    
    # Calculate performance metrics
    error = np.abs(temps - setpoints)
    mae = np.mean(error)
    settling_time = np.where(error < 1.0)[0]
    settling = len(settling_time) if len(settling_time) > 0 else 200
    
    # Fitness: minimize error and settling time
    fitness = 1.0 / (1.0 + mae + settling / 200.0)
    return fitness


def main():
    """Run control system example."""
    print("=" * 60)
    print("Control System Example - Temperature Controller")
    print("=" * 60)
    
    # Example 1: Manual Fuzzy Controller
    print("\n1. Creating manual fuzzy controller...")
    manual_controller = create_fuzzy_controller()
    print(f"   {manual_controller}")
    
    # Visualize membership functions
    print("\n2. Visualizing membership functions...")
    Visualizer.plot_membership_functions(
        manual_controller.input_variables["error"],
        title="Temperature Error Membership Functions"
    )
    
    Visualizer.plot_membership_functions(
        manual_controller.output_variables["power"],
        title="Heater Power Membership Functions"
    )
    
    # Simulate manual controller
    print("\n3. Simulating manual fuzzy controller...")
    time, temps, setpoints, control = simulate_control(manual_controller)
    
    manual_fitness = evaluate_controller(manual_controller)
    print(f"   Manual controller fitness: {manual_fitness:.4f}")
    
    # Visualize response
    print("\n4. Visualizing control response...")
    Visualizer.plot_control_response(
        time, setpoints, temps, control,
        title="Manual Fuzzy Controller Response"
    )
    
    # Example 2: GA-Optimized Fuzzy Controller
    print("\n5. Optimizing fuzzy controller with GA...")
    
    ga_fuzzy = GAFuzzy(
        input_ranges={"error": (-20, 20)},
        output_ranges={"power": (0, 100)},
        n_mf_per_variable=3,
        population_size=30,
        generations=20
    )
    
    # Optimize
    ga_fuzzy.optimize(
        control_function=evaluate_controller,
        crossover_rate=0.8,
        mutation_rate=0.1,
        verbose=True
    )
    
    # Get optimization history
    history = ga_fuzzy.get_optimization_history()
    
    print("\n6. Visualizing GA optimization...")
    Visualizer.plot_ga_fitness_evolution(
        history,
        title="GA Optimization of Fuzzy Controller"
    )
    
    # Simulate optimized controller
    print("\n7. Simulating GA-optimized controller...")
    time_opt, temps_opt, setpoints_opt, control_opt = simulate_control(
        ga_fuzzy.fuzzy_system
    )
    
    optimized_fitness = evaluate_controller(ga_fuzzy.fuzzy_system)
    print(f"   Optimized controller fitness: {optimized_fitness:.4f}")
    print(f"   Improvement: {(optimized_fitness - manual_fitness) / manual_fitness * 100:.2f}%")
    
    # Visualize optimized response
    print("\n8. Visualizing optimized control response...")
    Visualizer.plot_control_response(
        time_opt, setpoints_opt, temps_opt, control_opt,
        title="GA-Optimized Fuzzy Controller Response"
    )
    
    # Compare controllers
    print("\n9. Performance Comparison:")
    print(f"   Manual Controller:")
    manual_error = np.mean(np.abs(temps - setpoints))
    print(f"     Mean Absolute Error: {manual_error:.2f}°C")
    
    print(f"   GA-Optimized Controller:")
    opt_error = np.mean(np.abs(temps_opt - setpoints_opt))
    print(f"     Mean Absolute Error: {opt_error:.2f}°C")
    print(f"     Error Reduction: {(manual_error - opt_error) / manual_error * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("Control System Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
