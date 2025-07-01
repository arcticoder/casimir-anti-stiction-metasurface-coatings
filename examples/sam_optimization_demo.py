"""
Simple SAM Optimization Example

This example demonstrates how to optimize self-assembled monolayer (SAM) parameters
for anti-stiction applications using the work of adhesion calculation.

Target: Work of adhesion ≤ 10 mJ/m²
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_work_of_adhesion(gamma_sl, gamma_sv, gamma_lv, contact_angle):
    """
    Calculate work of adhesion using Young-Dupré equation
    
    W_adhesion = γ_SL - γ_SV - γ_LV*cos(θ)
    
    Parameters:
    - gamma_sl: solid-liquid interface energy (mJ/m²)
    - gamma_sv: solid-vapor interface energy (mJ/m²)
    - gamma_lv: liquid-vapor interface energy (mJ/m²)
    - contact_angle: contact angle (degrees)
    
    Returns:
    - work_adhesion: work of adhesion (mJ/m²)
    """
    theta_rad = np.radians(contact_angle)
    work_adhesion = gamma_sl - gamma_sv - gamma_lv * np.cos(theta_rad)
    return work_adhesion

def optimize_sam_parameters():
    """Optimize SAM parameters for minimum work of adhesion"""
    
    # SAM configurations with different chain lengths
    sam_configs = {
        'C8_thiol': {
            'gamma_sl': 25.0,
            'gamma_sv': 45.0,
            'contact_angle': 95
        },
        'C12_thiol': {
            'gamma_sl': 20.0,
            'gamma_sv': 48.0,
            'contact_angle': 105
        },
        'C18_thiol': {
            'gamma_sl': 18.0,
            'gamma_sv': 50.0,
            'contact_angle': 110
        },
        'fluorinated_SAM': {
            'gamma_sl': 15.0,
            'gamma_sv': 55.0,
            'contact_angle': 120
        }
    }
    
    gamma_lv = 72.8  # Water surface tension (mJ/m²)
    target_adhesion = 10.0  # Target maximum (mJ/m²)
    
    print("SAM Optimization for Anti-Stiction Applications")
    print("=" * 50)
    print(f"Target: Work of adhesion ≤ {target_adhesion} mJ/m²")
    print()
    
    results = {}
    
    for sam_name, params in sam_configs.items():
        work_adhesion = calculate_work_of_adhesion(
            params['gamma_sl'],
            params['gamma_sv'],
            gamma_lv,
            params['contact_angle']
        )
        
        results[sam_name] = work_adhesion
        
        meets_target = work_adhesion <= target_adhesion
        status = "✅ PASS" if meets_target else "❌ FAIL"
        
        print(f"{sam_name.upper()}:")
        print(f"  Work of Adhesion: {work_adhesion:.2f} mJ/m²")
        print(f"  Contact Angle: {params['contact_angle']}°")
        print(f"  Target Compliance: {status}")
        print()
    
    # Find optimal SAM
    best_sam = min(results.keys(), key=lambda k: results[k])
    best_adhesion = results[best_sam]
    
    print("OPTIMIZATION RESULTS:")
    print(f"  Optimal SAM: {best_sam.upper()}")
    print(f"  Best Work of Adhesion: {best_adhesion:.2f} mJ/m²")
    print(f"  Target Achievement: {((target_adhesion - best_adhesion)/target_adhesion)*100:.1f}% margin")
    
    return results

def plot_contact_angle_optimization():
    """Plot work of adhesion vs contact angle for optimization"""
    
    contact_angles = np.linspace(60, 140, 100)
    
    # Typical SAM surface energies
    gamma_sl = 18.0  # C18 thiol (mJ/m²)
    gamma_sv = 50.0  # C18 thiol (mJ/m²)
    gamma_lv = 72.8  # Water (mJ/m²)
    
    work_adhesions = []
    
    for angle in contact_angles:
        work_adhesion = calculate_work_of_adhesion(gamma_sl, gamma_sv, gamma_lv, angle)
        work_adhesions.append(work_adhesion)
    
    plt.figure(figsize=(10, 6))
    plt.plot(contact_angles, work_adhesions, 'b-', linewidth=2, label='Work of Adhesion')
    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Target: 10 mJ/m²')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Zero Adhesion')
    
    # Highlight optimal region
    optimal_mask = np.array(work_adhesions) <= 10.0
    if np.any(optimal_mask):
        plt.fill_between(contact_angles, work_adhesions, 10.0, 
                        where=optimal_mask, alpha=0.3, color='green', 
                        label='Acceptable Region')
    
    plt.xlabel('Contact Angle (degrees)')
    plt.ylabel('Work of Adhesion (mJ/m²)')
    plt.title('Work of Adhesion vs Contact Angle\n(C18 Thiol SAM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find optimal angle
    optimal_idx = np.argmin(np.abs(np.array(work_adhesions)))
    optimal_angle = contact_angles[optimal_idx]
    optimal_adhesion = work_adhesions[optimal_idx]
    
    plt.plot(optimal_angle, optimal_adhesion, 'ro', markersize=8, 
             label=f'Optimal: {optimal_angle:.1f}°, {optimal_adhesion:.2f} mJ/m²')
    
    plt.tight_layout()
    plt.savefig('sam_contact_angle_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Optimal contact angle: {optimal_angle:.1f}°")
    print(f"Resulting work of adhesion: {optimal_adhesion:.2f} mJ/m²")

if __name__ == "__main__":
    # Run SAM optimization
    results = optimize_sam_parameters()
    
    # Generate contact angle optimization plot
    print("\nGenerating contact angle optimization plot...")
    plot_contact_angle_optimization()
    
    print("\nSAM optimization complete!")
    print("Generated: sam_contact_angle_optimization.png")
