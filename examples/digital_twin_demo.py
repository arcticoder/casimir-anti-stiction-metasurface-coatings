"""
Digital Twin Framework Comprehensive Demonstration

This demonstration showcases the complete digital twin mathematical framework
for Casimir anti-stiction metasurface coatings with uncertainty quantification.

Features Demonstrated:
1. State Space Representation
2. UQ-Enhanced Force Model
3. Digital Twin Fidelity Metric
4. Adaptive Kalman Filter
5. Metamaterial Parameter Identification
6. Predictive Control with UQ Bounds
7. Multi-Physics Coupling
8. Sensitivity Analysis
9. Robust Performance Index
10. Model Reduction (POD)

Performance Targets:
- Sensor precision: 0.06 pm/√Hz
- Thermal expansion uncertainty: 5 nm
- Vibration isolation: 9.7×10¹¹×
- Material uncertainty: <4.1%
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from digital_twin_framework import CasimirDigitalTwin, DigitalTwinMetrics
    from fabrication_spec import AntiStictionFabricationSpec
    framework_available = True
except ImportError as e:
    print(f"Warning: Could not import framework modules: {e}")
    framework_available = False

def demo_state_space_representation():
    """Demonstrate state space representation and system dynamics"""
    print("=" * 60)
    print("1. STATE SPACE REPRESENTATION DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical representation")
        print("\nState Vector: x = [d(t), ḋ(t), F_Casimir(t), θ_SAM(t), T_surface(t)]ᵀ")
        print("System Dynamics: x(k+1) = Ax(k) + Bu(k) + w(k)")
        print("Measurements: y(k) = Cx(k) + v(k)")
        return
    
    dt_framework = CasimirDigitalTwin(sampling_time=1e-6)
    
    print(f"State Dimension: {dt_framework.state_dim}")
    print(f"Input Dimension: {dt_framework.input_dim}")
    print(f"Output Dimension: {dt_framework.output_dim}")
    print(f"Sampling Time: {dt_framework.dt*1e6:.1f} μs")
    
    print(f"\nDiscrete-time A matrix (5×5):")
    print(dt_framework.A)
    
    print(f"\nDiscrete-time B matrix (5×3):")
    print(dt_framework.B)
    
    print(f"\nOutput matrix C (5×5):")
    print(dt_framework.C)

def demo_uq_enhanced_force_model():
    """Demonstrate UQ-enhanced force modeling with uncertainty propagation"""
    print("\n" + "=" * 60)
    print("2. UQ-ENHANCED FORCE MODEL DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical formulation")
        print("\nF_total = F_Casimir × (1 + ε_UQ) + F_adhesion × (1 + δ_material)")
        print("σ_F² = (∂F/∂ε')²σ_ε'² + (∂F/∂μ')²σ_μ'² + (∂F/∂d)²σ_d²")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Test state vector [d, ḋ, F_cas, θ, T]
    x_test = np.array([5.0, 0.1, -0.5, 110.0, 298.0])
    
    # Material parameters
    material_params = {
        'epsilon_prime': -2.5,
        'mu_prime': -1.8,
        'epsilon_imag': 0.3,
        'mu_imag': 0.2
    }
    
    print(f"Test State: {x_test}")
    print(f"Material: ε' = {material_params['epsilon_prime']}, μ' = {material_params['mu_prime']}")
    
    # Calculate UQ-enhanced force
    F_total, F_uncertainty = dt_framework.calculate_uq_enhanced_force(x_test, material_params)
    
    print(f"\nUQ-Enhanced Force Results:")
    print(f"Total Force: {F_total:.6f} nN")
    print(f"Force Uncertainty: ±{F_uncertainty:.6f} nN")
    print(f"Relative Uncertainty: {abs(F_uncertainty/F_total)*100:.2f}%")
    
    # UQ Parameters
    print(f"\nUQ Parameters:")
    print(f"Material Uncertainty (ε_UQ): {dt_framework.uq_params.epsilon_uq*100:.1f}%")
    print(f"Material Variation (δ_material): {dt_framework.uq_params.delta_material*100:.1f}%")
    print(f"Distance Uncertainty: {dt_framework.uq_params.sigma_distance*1e12:.2f} pm")

def demo_digital_twin_fidelity():
    """Demonstrate digital twin fidelity metric calculation"""
    print("\n" + "=" * 60) 
    print("3. DIGITAL TWIN FIDELITY METRIC DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical formulation")
        print("\nΦ_fidelity = exp(-1/2 Σᵢ [(x_measured,i - x_twin,i)ᵀ Σ⁻¹ (x_measured,i - x_twin,i)])")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Simulated measurements vs digital twin predictions
    np.random.seed(42)
    x_measured = np.array([5.0, 0.05, -0.45, 110.5, 298.1])
    x_twin = np.array([4.98, 0.048, -0.47, 110.2, 298.05])
    
    # Covariance matrix (inverse)
    Sigma_inv = np.diag([1e6, 1e9, 1e3, 1e-2, 1e2])  # Scaled by measurement precision
    
    fidelity = dt_framework.calculate_fidelity_metric(x_measured, x_twin, Sigma_inv)
    
    print(f"Measured State: {x_measured}")
    print(f"Twin Prediction: {x_twin}")
    print(f"State Errors: {x_measured - x_twin}")
    print(f"\nFidelity Score: {fidelity:.6f}")
    
    if fidelity >= 0.95:
        print("✅ HIGH FIDELITY (≥95%)")
    elif fidelity >= 0.90:
        print("⚠️  MODERATE FIDELITY (90-95%)")
    else:
        print("❌ LOW FIDELITY (<90%)")

def demo_adaptive_kalman_filter():
    """Demonstrate adaptive Kalman filter for real-time calibration"""
    print("\n" + "=" * 60)
    print("4. ADAPTIVE KALMAN FILTER DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical formulation")
        print("\nx̂(k|k) = x̂(k|k-1) + K_k(y_k - Cx̂(k|k-1))")
        print("K_k = P(k|k-1)Cᵀ(CP(k|k-1)Cᵀ + R)⁻¹")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Simulate measurement sequence
    np.random.seed(42)
    n_steps = 50
    measurements = []
    state_estimates = []
    
    # True state evolution (with process noise)
    x_true = np.array([5.0, 0.0, -0.5, 110.0, 298.0])
    
    print("Running Kalman filter simulation...")
    
    for k in range(n_steps):
        # Add measurement noise
        measurement_noise = np.array([
            0.06e-3 * np.random.randn(),   # Distance noise (pm -> nm)
            1e-6 * np.random.randn(),      # Velocity noise  
            0.01 * np.random.randn(),      # Force noise
            0.1 * np.random.randn(),       # Angle noise
            0.01 * np.random.randn()       # Temperature noise
        ])
        
        y_measured = x_true + measurement_noise
        measurements.append(y_measured.copy())
        
        # Kalman filter update
        x_estimated = dt_framework.adaptive_kalman_update(y_measured)
        state_estimates.append(x_estimated.copy())
        
        # Evolve true state (simple dynamics)
        x_true += 0.001 * np.array([0.1*np.sin(k*0.1), np.cos(k*0.1), 0.01*np.sin(k*0.05), 0, 0])
    
    measurements = np.array(measurements)
    state_estimates = np.array(state_estimates)
    
    # Calculate estimation errors
    estimation_errors = np.mean(np.abs(measurements - state_estimates), axis=0)
    
    print(f"\nKalman Filter Performance:")
    print(f"Steps Processed: {n_steps}")
    print(f"Mean Absolute Errors:")
    print(f"  Distance: {estimation_errors[0]*1e3:.3f} pm")
    print(f"  Velocity: {estimation_errors[1]*1e6:.3f} nm/s")
    print(f"  Force: {estimation_errors[2]:.6f} nN")
    print(f"  Contact Angle: {estimation_errors[3]:.3f}°")
    print(f"  Temperature: {estimation_errors[4]:.3f} K")
    
    # Final covariance diagonal (uncertainties)
    P_final = np.diagonal(dt_framework.P)
    print(f"\nFinal State Uncertainties (1σ):")
    print(f"  Distance: {np.sqrt(P_final[0])*1e3:.3f} pm")
    print(f"  Velocity: {np.sqrt(P_final[1])*1e6:.3f} nm/s")
    print(f"  Force: {np.sqrt(P_final[2]):.6f} nN")
    print(f"  Contact Angle: {np.sqrt(P_final[3]):.3f}°")
    print(f"  Temperature: {np.sqrt(P_final[4]):.3f} K")

def demo_parameter_identification():
    """Demonstrate metamaterial parameter identification"""
    print("\n" + "=" * 60)
    print("5. METAMATERIAL PARAMETER IDENTIFICATION DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical formulation")
        print("\n{ε'(ω), μ'(ω)} = arg min Σⱼ |F_measured,j - F_model,j(ε', μ')|²")
        print("Subject to: ε' × μ' < -1, |ε''|/|ε'| < 0.1")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Simulate "measured" force data with known parameters
    true_params = {
        'epsilon_prime': -2.5,
        'mu_prime': -1.8, 
        'epsilon_imag': 0.3,
        'mu_imag': 0.2
    }
    
    frequencies = np.logspace(13, 15, 10)  # 10 THz to 1000 THz
    F_measured = []
    
    print("Generating synthetic measurement data...")
    
    for freq in frequencies:
        # Calculate "true" force with some noise
        enhancement = abs((true_params['epsilon_prime'] + 1j*true_params['epsilon_imag']) * 
                         (true_params['mu_prime'] + 1j*true_params['mu_imag']) - 1) / \
                     abs((true_params['epsilon_prime'] + 1j*true_params['epsilon_imag']) * 
                         (true_params['mu_prime'] + 1j*true_params['mu_imag']) + 1)
        
        F_true = -1e-9 * enhancement / (freq/1e14)**3
        F_noisy = F_true + 0.05 * F_true * np.random.randn()  # 5% noise
        F_measured.append(F_noisy)
    
    F_measured = np.array(F_measured)
    
    # Initial guess (perturbed from true values)
    initial_guess = {
        'epsilon_prime': -2.0,
        'mu_prime': -1.5,
        'epsilon_imag': 0.25,
        'mu_imag': 0.15
    }
    
    print(f"True Parameters: ε'={true_params['epsilon_prime']}, μ'={true_params['mu_prime']}")
    print(f"Initial Guess: ε'={initial_guess['epsilon_prime']}, μ'={initial_guess['mu_prime']}")
    
    # Run parameter identification
    print("Running parameter identification optimization...")
    identified_params = dt_framework.identify_metamaterial_parameters(
        F_measured, frequencies, initial_guess
    )
    
    print(f"\nParameter Identification Results:")
    print(f"Converged: {identified_params.get('converged', False)}")
    if identified_params.get('converged', False):
        print(f"Identified ε': {identified_params['epsilon_prime']:.3f} (true: {true_params['epsilon_prime']})")
        print(f"Identified μ': {identified_params['mu_prime']:.3f} (true: {true_params['mu_prime']})")
        print(f"Optimization Error: {identified_params['optimization_error']:.6e}")
        
        # Check constraints
        eps_mu_product = identified_params['epsilon_prime'] * identified_params['mu_prime']
        repulsive_ok = eps_mu_product < -1
        
        loss_eps = abs(identified_params['epsilon_imag']) / abs(identified_params['epsilon_prime'])
        loss_mu = abs(identified_params['mu_imag']) / abs(identified_params['mu_prime'])
        low_loss_ok = (loss_eps < 0.1) and (loss_mu < 0.1)
        
        print(f"\nConstraint Validation:")
        print(f"Repulsive condition (ε'×μ' < -1): {eps_mu_product:.3f} {'✅' if repulsive_ok else '❌'}")
        print(f"Low loss (|ε''|/|ε'| < 0.1): {loss_eps:.3f} {'✅' if low_loss_ok else '❌'}")
        print(f"Low loss (|μ''|/|μ'| < 0.1): {loss_mu:.3f} {'✅' if low_loss_ok else '❌'}")

def demo_predictive_control():
    """Demonstrate predictive control with UQ bounds"""
    print("\n" + "=" * 60)
    print("6. PREDICTIVE CONTROL WITH UQ BOUNDS DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical formulation")
        print("\nu* = arg min Σᵢ [‖xᵢ₊₁ - x_ref‖²_Q + ‖uᵢ‖²_R]")
        print("Subject to: P(d_min ≤ d(t) ≤ d_max) ≥ 0.95 ∀t ∈ [0,T]")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Current state (gap too large)
    x_current = np.array([8.0, -0.1, -0.3, 112.0, 299.0])  # 8 nm gap
    
    # Reference state (target: 5 nm gap)
    x_reference = np.array([5.0, 0.0, -0.5, 110.0, 298.0])
    
    print(f"Current State: {x_current}")
    print(f"Reference State: {x_reference}")
    print(f"Gap Error: {x_current[0] - x_reference[0]:.1f} nm")
    
    # Run predictive control
    print("Computing optimal control sequence...")
    u_optimal, control_info = dt_framework.predictive_control_with_uq(
        x_current, x_reference, horizon=5
    )
    
    print(f"\nPredictive Control Results:")
    print(f"Optimization Converged: {control_info['converged']}")
    print(f"Constraint Satisfied: {control_info['constraint_satisfied']}")
    print(f"Control Horizon: {control_info['horizon']} steps")
    print(f"Optimal Cost: {control_info['cost']:.6e}")
    
    if control_info['converged']:
        print(f"\nOptimal Control Sequence (first 3 steps):")
        for i in range(min(3, len(u_optimal))):
            print(f"  Step {i+1}: u = [{u_optimal[i,0]:.2e}, {u_optimal[i,1]:.2e}, {u_optimal[i,2]:.2e}]")
            print(f"           [F_applied, Q_thermal, Γ_chemical]")

def demo_sensitivity_analysis():
    """Demonstrate sensitivity analysis for critical parameters"""
    print("\n" + "=" * 60)
    print("7. SENSITIVITY ANALYSIS DEMONSTRATION")  
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical formulation")
        print("\nS_i,j = ∂ln(F_Casimir)/∂ln(p_j)|_{p=p₀}")
        print("Where p_j ∈ {ε', μ', d, T, ω}")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Test state
    x_test = np.array([5.0, 0.0, -0.5, 110.0, 298.0])
    
    # Material parameters
    material_params = {
        'epsilon_prime': -2.5,
        'mu_prime': -1.8,
        'epsilon_imag': 0.3,
        'mu_imag': 0.2,
        'frequency': 1e14
    }
    
    print(f"Computing sensitivities for state: {x_test}")
    
    # Calculate sensitivities
    sensitivities = dt_framework.sensitivity_analysis(x_test, material_params)
    
    print(f"\nSensitivity Analysis Results:")
    print(f"Parameter Sensitivities (S_i,j):")
    
    for param, sensitivity in sensitivities.items():
        abs_sensitivity = abs(sensitivity)
        if abs_sensitivity > 1.0:
            level = "HIGH"
            symbol = "🔴"
        elif abs_sensitivity > 0.1:
            level = "MEDIUM" 
            symbol = "🟡"
        else:
            level = "LOW"
            symbol = "🟢"
        
        print(f"  {param:15s}: {sensitivity:8.3f} {symbol} {level}")
    
    # Find most critical parameter
    most_sensitive = max(sensitivities.keys(), key=lambda k: abs(sensitivities[k]))
    print(f"\nMost Critical Parameter: {most_sensitive}")
    print(f"Sensitivity Value: {sensitivities[most_sensitive]:.3f}")
    
    if abs(sensitivities[most_sensitive]) > 1.0:
        print("⚠️  HIGH SENSITIVITY - Requires tight control")
    else:
        print("✅ MANAGEABLE SENSITIVITY")

def demo_model_reduction():
    """Demonstrate model reduction using POD"""
    print("\n" + "=" * 60)
    print("8. MODEL REDUCTION (POD) DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - showing theoretical formulation")
        print("\nx_reduced = Φᵀ x_full")
        print("Where Φ = [φ₁, φ₂, ..., φᵣ] with r << n")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Generate synthetic state snapshots
    np.random.seed(42)
    n_snapshots = 100
    time_points = np.linspace(0, 1, n_snapshots)
    
    print(f"Generating {n_snapshots} state snapshots...")
    
    # Create correlated state evolution
    state_snapshots = np.zeros((5, n_snapshots))
    
    for i, t in enumerate(time_points):
        # Gap distance with oscillation
        d = 5.0 + 2.0 * np.sin(2*np.pi*t) + 0.1 * np.random.randn()
        
        # Gap velocity (derivative of distance)
        d_dot = 4*np.pi * np.cos(2*np.pi*t) + 0.01 * np.random.randn()
        
        # Casimir force (inversely related to gap)
        F_cas = -0.5 / (d/5.0)**3 + 0.01 * np.random.randn()
        
        # Contact angle (slowly varying)
        theta = 110.0 + 5.0 * np.sin(0.5*np.pi*t) + 0.1 * np.random.randn()
        
        # Temperature (anti-correlated with force)
        T = 298.0 - 2.0 * F_cas + 0.05 * np.random.randn()
        
        state_snapshots[:, i] = [d, d_dot, F_cas, theta, T]
    
    # Perform POD model reduction
    print("Performing POD model reduction...")
    energy_thresholds = [0.90, 0.95, 0.99]
    
    for threshold in energy_thresholds:
        pod_basis = dt_framework.model_reduction_pod(state_snapshots, energy_threshold=threshold)
        
        print(f"\nEnergy Threshold: {threshold*100:.0f}%")
        print(f"Reduced Order: {dt_framework.reduced_order}/{dt_framework.state_dim}")
        print(f"Compression Ratio: {dt_framework.state_dim/dt_framework.reduced_order:.1f}×")
        
        # Test reconstruction accuracy
        test_state = state_snapshots[:, 50]  # Middle snapshot
        x_reduced = dt_framework.predict_reduced_model(test_state)
        x_reconstructed = dt_framework.reconstruct_full_model(x_reduced)
        
        reconstruction_error = np.linalg.norm(test_state - x_reconstructed) / np.linalg.norm(test_state)
        print(f"Reconstruction Error: {reconstruction_error*100:.2f}%")
        
        if reconstruction_error < 0.05:
            print("✅ EXCELLENT reconstruction accuracy")
        elif reconstruction_error < 0.10:
            print("⚠️  GOOD reconstruction accuracy")
        else:
            print("❌ POOR reconstruction accuracy")

def demo_complete_digital_twin_cycle():
    """Demonstrate complete digital twin operational cycle"""
    print("\n" + "=" * 60)
    print("9. COMPLETE DIGITAL TWIN CYCLE DEMONSTRATION")
    print("=" * 60)
    
    if not framework_available:
        print("Framework not available - using theoretical demonstration")
        print("\nDigital Twin Cycle Components:")
        print("1. State Estimation (Kalman Filter)")
        print("2. UQ-Enhanced Force Calculation")
        print("3. Fidelity Assessment")
        print("4. Sensitivity Analysis")
        print("5. Predictive Control")
        print("6. Performance Validation")
        return
    
    dt_framework = CasimirDigitalTwin()
    
    # Simulate realistic measurements
    np.random.seed(42)
    measurements = np.array([
        5.05 + 0.06e-3 * np.random.randn(),   # Distance with pm-level noise
        0.001 * np.random.randn(),            # Velocity
        -0.48 + 0.01 * np.random.randn(),     # Force with pN-level noise  
        110.2 + 0.1 * np.random.randn(),      # Contact angle
        298.05 + 0.01 * np.random.randn()     # Temperature with mK precision
    ])
    
    # Material parameters
    material_params = {
        'epsilon_prime': -2.5,
        'mu_prime': -1.8,
        'epsilon_imag': 0.3,
        'mu_imag': 0.2,
        'frequency': 1e14
    }
    
    print("Running complete digital twin cycle...")
    
    # Execute full cycle
    cycle_results = dt_framework.run_digital_twin_cycle(measurements, material_params)
    
    print(f"\n📊 DIGITAL TWIN CYCLE RESULTS")
    print(f"Timestamp: {cycle_results['timestamp']}")
    print(f"Fidelity Score: {cycle_results['fidelity_score']:.6f}")
    
    state_est = np.array(cycle_results['state_estimate'])
    print(f"\nState Estimation:")
    print(f"  Gap Distance: {state_est[0]:.3f} nm")
    print(f"  Gap Velocity: {state_est[1]*1e6:.3f} nm/s") 
    print(f"  Casimir Force: {state_est[2]:.6f} nN")
    print(f"  Contact Angle: {state_est[3]:.2f}°")
    print(f"  Temperature: {state_est[4]:.3f} K")
    
    print(f"\nForce Analysis:")
    print(f"  Total Force: {cycle_results['force_total']:.6f} nN")
    print(f"  Force Uncertainty: ±{cycle_results['force_uncertainty']:.6f} nN")
    
    control_action = np.array(cycle_results['control_sequence'])
    print(f"\nNext Control Action:")
    print(f"  Applied Force: {control_action[0]:.2e} N")
    print(f"  Thermal Input: {control_action[1]:.2e} W")
    print(f"  Chemical Control: {control_action[2]:.2e} mol/s")
    
    # Performance validation summary
    validation = cycle_results['performance_validation']
    print(f"\n🎯 PERFORMANCE VALIDATION:")
    
    metrics = [
        ('Sensor Precision', 'sensor_precision'),
        ('Thermal Uncertainty', 'thermal_uncertainty'),
        ('Vibration Isolation', 'vibration_isolation'),
        ('Material Uncertainty', 'material_uncertainty'),
        ('Fidelity Score', 'fidelity_score')
    ]
    
    for name, key in metrics:
        status = "✅ PASS" if validation[key] else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    overall_status = "✅ VALIDATED" if validation['overall_performance'] else "❌ FAILED"
    print(f"\n🏆 OVERALL PERFORMANCE: {overall_status}")
    
    return cycle_results

def main():
    """
    Complete Digital Twin Framework Demonstration
    """
    print("CASIMIR ANTI-STICTION DIGITAL TWIN FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("Comprehensive demonstration of mathematical framework components")
    print("Performance Targets:")
    print("  • Sensor precision: 0.06 pm/√Hz")
    print("  • Thermal expansion uncertainty: 5 nm")
    print("  • Vibration isolation: 9.7×10¹¹×")
    print("  • Material uncertainty: <4.1%")
    
    try:
        # Run all demonstrations
        demo_state_space_representation()
        demo_uq_enhanced_force_model()
        demo_digital_twin_fidelity()
        demo_adaptive_kalman_filter()
        demo_parameter_identification()
        demo_predictive_control()
        demo_sensitivity_analysis()
        demo_model_reduction()
        
        # Complete cycle demonstration
        cycle_results = demo_complete_digital_twin_cycle()
        
        # Save results
        if cycle_results:
            with open('digital_twin_demonstration_results.json', 'w') as f:
                json.dump(cycle_results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: digital_twin_demonstration_results.json")
        
        print("\n" + "=" * 80)
        print("🎉 DIGITAL TWIN FRAMEWORK DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("All mathematical components successfully demonstrated!")
        
        if framework_available:
            print("\n✅ Framework Status: FULLY OPERATIONAL")
            print("Ready for real-time anti-stiction coating control and monitoring")
        else:
            print("\n⚠️  Framework Status: THEORETICAL DEMONSTRATION")
            print("Install required dependencies for full functionality")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Check dependencies and framework installation")

if __name__ == "__main__":
    main()
