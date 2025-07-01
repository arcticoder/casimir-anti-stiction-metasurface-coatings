"""
Anti-Stiction Metasurface Coating Design and Validation Demo

This demonstration script showcases the complete anti-stiction coating design process,
from mathematical optimization to fabrication validation.

Key Features:
- Metamaterial enhancement factor calculation
- Casimir force analysis for repulsive configurations
- Pull-in gap optimization
- SAM work of adhesion control
- Performance validation against target specifications

Target Achievements:
- Static pull-in gap: ≥5 nm
- Work of adhesion: ≤10 mJ/m²
- Enhancement factor: ≥100×
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src" / "prototype"))

from fabrication_spec import (
    AntiStictionFabricationSpec, 
    CoatingTechnology, 
    MetamaterialType,
    MaterialProperties
)

def demo_metamaterial_enhancement():
    """Demonstrate metamaterial enhancement factor calculations"""
    print("=" * 60)
    print("METAMATERIAL ENHANCEMENT FACTOR DEMONSTRATION")
    print("=" * 60)
    
    fab_spec = AntiStictionFabricationSpec()
    
    # Test different metamaterial types
    materials = fab_spec.material_database
    frequencies = np.logspace(12, 15, 100)  # 1 THz to 1000 THz
    
    plt.figure(figsize=(12, 8))
    
    for material_name, material in materials.items():
        enhancements = []
        
        for freq in frequencies:
            if material.frequency_range[0] <= freq <= material.frequency_range[1]:
                enhancement = fab_spec.calculate_enhancement_factor(material, freq)
                enhancements.append(enhancement)
            else:
                enhancements.append(np.nan)
        
        plt.loglog(frequencies/1e12, enhancements, label=material_name.replace('_', ' ').title(), linewidth=2)
        
        # Print key statistics
        valid_enhancements = [e for e in enhancements if not np.isnan(e)]
        if valid_enhancements:
            max_enhancement = max(valid_enhancements)
            print(f"{material_name.upper()}:")
            print(f"  Maximum Enhancement: {max_enhancement:.1f}×")
            print(f"  Frequency Range: {material.frequency_range[0]/1e12:.1f}-{material.frequency_range[1]/1e12:.1f} THz")
            print(f"  ε' = {material.epsilon_real:.1f}, μ' = {material.mu_real:.1f}")
            
            # Check if repulsive (both ε and μ negative)
            if material.epsilon_real < 0 and material.mu_real < 0:
                print(f"  Status: ✅ REPULSIVE (ε < 0, μ < 0)")
            else:
                print(f"  Status: ❌ ATTRACTIVE")
            print()
    
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Enhancement Factor A_meta')
    plt.title('Metamaterial Enhancement Factor vs Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Target: 100× minimum')
    plt.ylim(1, 10000)
    
    plt.tight_layout()
    plt.savefig('metamaterial_enhancement_analysis.png', dpi=300, bbox_inches='tight')
    print("Enhancement factor analysis saved to: metamaterial_enhancement_analysis.png")

def demo_casimir_force_calculation():
    """Demonstrate Casimir force calculation for different separations"""
    print("\n" + "=" * 60)
    print("CASIMIR FORCE CALCULATION DEMONSTRATION")
    print("=" * 60)
    
    fab_spec = AntiStictionFabricationSpec()
    
    # Test hyperbolic metamaterial (most promising for anti-stiction)
    material = fab_spec.material_database['hyperbolic_metamaterial_1']
    
    # Range of separations from 1 nm to 100 nm
    separations = np.logspace(0, 2, 50)  # 1 to 100 nm
    forces = []
    
    print("Calculating Casimir forces...")
    print(f"Material: Hyperbolic Metamaterial (ε'={material.epsilon_real}, μ'={material.mu_real})")
    
    for sep in separations:
        force = fab_spec.calculate_casimir_force(material, sep)
        forces.append(force)
        
        # Print key separation points
        if sep in [1, 5, 10, 50, 100]:
            force_nN = force * 1e9  # Convert to nN
            force_type = "REPULSIVE" if force < 0 else "ATTRACTIVE"
            print(f"  Separation: {sep:3.0f} nm → Force: {force_nN:8.3f} nN ({force_type})")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    forces_nN = np.array(forces) * 1e9  # Convert to nN
    
    plt.loglog(separations, np.abs(forces_nN), 'b-', linewidth=2, label='|Casimir Force|')
    
    # Indicate repulsive vs attractive regions
    repulsive_mask = np.array(forces) < 0
    attractive_mask = np.array(forces) > 0
    
    if np.any(repulsive_mask):
        plt.loglog(separations[repulsive_mask], np.abs(forces_nN[repulsive_mask]), 
                  'go', markersize=4, label='Repulsive Force')
    
    if np.any(attractive_mask):
        plt.loglog(separations[attractive_mask], np.abs(forces_nN[attractive_mask]), 
                  'ro', markersize=4, label='Attractive Force')
    
    plt.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Target: 5 nm pull-in gap')
    plt.xlabel('Separation Distance (nm)')
    plt.ylabel('|Casimir Force| (nN)')
    plt.title('Casimir Force vs Separation Distance\n(Hyperbolic Metamaterial)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('casimir_force_analysis.png', dpi=300, bbox_inches='tight')
    print("Casimir force analysis saved to: casimir_force_analysis.png")

def demo_pull_in_gap_optimization():
    """Demonstrate pull-in gap optimization"""
    print("\n" + "=" * 60)
    print("PULL-IN GAP OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    fab_spec = AntiStictionFabricationSpec()
    
    # Parameter ranges for optimization
    spring_constants = np.logspace(-6, -2, 50)  # 1 μN/m to 10 mN/m
    voltages = np.logspace(-2, 2, 50)          # 10 mV to 100 V
    
    target_gap = 5.0  # nm
    
    print(f"Target pull-in gap: {target_gap} nm")
    print("Optimizing spring constant and voltage parameters...")
    
    # Create parameter grid
    K_grid, V_grid = np.meshgrid(spring_constants, voltages)
    gap_grid = np.zeros_like(K_grid)
    
    for i in range(len(voltages)):
        for j in range(len(spring_constants)):
            gap = fab_spec.calculate_pull_in_gap(spring_constants[j], voltages[i])
            gap_grid[i, j] = gap
    
    # Find optimal parameters
    target_mask = np.abs(gap_grid - target_gap) < 0.5  # Within 0.5 nm of target
    
    if np.any(target_mask):
        optimal_indices = np.where(target_mask)
        optimal_k = spring_constants[optimal_indices[1][0]]
        optimal_v = voltages[optimal_indices[0][0]]
        optimal_gap = gap_grid[optimal_indices[0][0], optimal_indices[1][0]]
        
        print(f"\nOptimal Parameters Found:")
        print(f"  Spring Constant: {optimal_k*1e6:.2f} μN/m")
        print(f"  Voltage: {optimal_v:.3f} V")  
        print(f"  Resulting Gap: {optimal_gap:.2f} nm")
        print(f"  Target Achievement: ✅ SUCCESS")
    else:
        print(f"  Target Achievement: ❌ FAILED (no parameters found)")
    
    # Create contour plot
    plt.figure(figsize=(10, 8))
    
    levels = np.logspace(0, 3, 20)  # 1 nm to 1000 nm
    cs = plt.contourf(K_grid*1e6, V_grid, gap_grid, levels=levels, cmap='viridis')
    plt.colorbar(cs, label='Pull-in Gap (nm)')
    
    # Mark target gap contour
    target_contour = plt.contour(K_grid*1e6, V_grid, gap_grid, levels=[target_gap], 
                                colors='red', linewidths=2, linestyles='--')
    plt.clabel(target_contour, inline=True, fontsize=10, fmt='%d nm target')
    
    if np.any(target_mask):
        plt.plot(optimal_k*1e6, optimal_v, 'r*', markersize=15, label='Optimal Point')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Spring Constant (μN/m)')
    plt.ylabel('Voltage (V)')
    plt.title('Pull-in Gap Optimization Map')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pull_in_gap_optimization.png', dpi=300, bbox_inches='tight')
    print("Pull-in gap optimization saved to: pull_in_gap_optimization.png")

def demo_sam_work_of_adhesion():
    """Demonstrate SAM work of adhesion calculations"""
    print("\n" + "=" * 60)
    print("SAM WORK OF ADHESION DEMONSTRATION")  
    print("=" * 60)
    
    fab_spec = AntiStictionFabricationSpec()
    
    # Surface energy parameters for different SAM configurations
    sam_configurations = {
        'C8_thiol': {
            'gamma_sl': 25.0,    # mJ/m²
            'gamma_sv': 45.0,    # mJ/m²
            'gamma_lv': 72.8,    # mJ/m² (water)
            'contact_angle': 95   # degrees
        },
        'C12_thiol': {
            'gamma_sl': 20.0,
            'gamma_sv': 48.0,
            'gamma_lv': 72.8,
            'contact_angle': 105
        },
        'C18_thiol': {
            'gamma_sl': 18.0,
            'gamma_sv': 50.0,
            'gamma_lv': 72.8,
            'contact_angle': 110
        },
        'fluorinated_sam': {
            'gamma_sl': 15.0,
            'gamma_sv': 55.0,
            'gamma_lv': 72.8,
            'contact_angle': 120
        }
    }
    
    target_adhesion = 10.0  # mJ/m² maximum
    
    print(f"Target work of adhesion: ≤{target_adhesion} mJ/m²")
    print("\nSAM Configuration Analysis:")
    
    results = {}
    
    for sam_name, params in sam_configurations.items():
        work_adhesion = fab_spec.calculate_work_of_adhesion(
            params['gamma_sl'],
            params['gamma_sv'], 
            params['gamma_lv'],
            params['contact_angle']
        )
        
        results[sam_name] = work_adhesion
        
        meets_target = work_adhesion <= target_adhesion
        status = "✅ PASS" if meets_target else "❌ FAIL"
        
        print(f"\n  {sam_name.upper()}:")
        print(f"    Work of Adhesion: {work_adhesion:.2f} mJ/m²")
        print(f"    Contact Angle: {params['contact_angle']}°")
        print(f"    Target Compliance: {status}")
    
    # Find best SAM
    best_sam = min(results.keys(), key=lambda k: results[k])
    best_adhesion = results[best_sam]
    
    print(f"\n  OPTIMAL SAM: {best_sam.upper()}")
    print(f"  Best Work of Adhesion: {best_adhesion:.2f} mJ/m²")
    print(f"  Improvement over target: {((target_adhesion - best_adhesion)/target_adhesion)*100:.1f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    sam_names = list(results.keys())
    adhesions = list(results.values())
    colors = ['green' if w <= target_adhesion else 'red' for w in adhesions]
    
    bars = plt.bar(range(len(sam_names)), adhesions, color=colors, alpha=0.7)
    plt.axhline(y=target_adhesion, color='red', linestyle='--', alpha=0.8, 
                label=f'Target: {target_adhesion} mJ/m²')
    
    plt.xlabel('SAM Configuration')
    plt.ylabel('Work of Adhesion (mJ/m²)')
    plt.title('SAM Work of Adhesion Analysis')
    plt.xticks(range(len(sam_names)), [name.replace('_', ' ').title() for name in sam_names], 
               rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, adhesions)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sam_work_of_adhesion_analysis.png', dpi=300, bbox_inches='tight')
    print("SAM work of adhesion analysis saved to: sam_work_of_adhesion_analysis.png")

def demo_coating_optimization():
    """Demonstrate complete coating technology optimization"""
    print("\n" + "=" * 60)
    print("COMPLETE COATING TECHNOLOGY OPTIMIZATION")
    print("=" * 60)
    
    fab_spec = AntiStictionFabricationSpec()
    
    # Test all coating technologies
    technologies = [
        CoatingTechnology.SAM,
        CoatingTechnology.METAMATERIAL_SPACER,
        CoatingTechnology.DIELECTRIC_STACK
    ]
    
    optimization_results = {}
    
    for tech in technologies:
        print(f"\nOptimizing {tech.value.upper()} Technology...")
        
        result = fab_spec.optimize_coating_design(tech, target_gap=5.0)
        validation = fab_spec.validate_performance(result)
        
        optimization_results[tech] = {
            'optimization': result,
            'validation': validation
        }
        
        # Print summary
        overall_status = "✅ VALIDATED" if validation['overall_validation'] else "❌ FAILED"
        print(f"  Optimization Status: {overall_status}")
        
        if 'enhancement_factor' in result.get('performance', {}):
            enhancement = result['performance']['enhancement_factor']
            print(f"  Enhancement Factor: {enhancement:.1f}×")
        
        if 'work_of_adhesion_mJ_per_m2' in result.get('performance', {}):
            adhesion = result['performance']['work_of_adhesion_mJ_per_m2']
            print(f"  Work of Adhesion: {adhesion:.2f} mJ/m²")
    
    # Summary comparison
    print("\n" + "=" * 40)
    print("TECHNOLOGY COMPARISON SUMMARY")
    print("=" * 40)
    
    for tech, data in optimization_results.items():
        validation = data['validation']
        status = "✅ VALIDATED" if validation['overall_validation'] else "❌ FAILED"
        print(f"{tech.value.upper():25} {status}")
    
    # Find best technology
    validated_techs = [tech for tech, data in optimization_results.items() 
                      if data['validation']['overall_validation']]
    
    if validated_techs:
        print(f"\nRECOMMENDED TECHNOLOGIES:")
        for tech in validated_techs:
            print(f"  • {tech.value.upper()}")
        
        # Generate full report for best technology
        best_tech = validated_techs[0]  # Could implement scoring here
        print(f"\nDETAILED REPORT FOR {best_tech.value.upper()}:")
        print("-" * 50)
        report = fab_spec.generate_fabrication_report(best_tech)
        print(report)
        
        # Save report
        report_filename = f"anti_stiction_{best_tech.value}_detailed_report.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"Detailed report saved to: {report_filename}")
    else:
        print("\n❌ NO TECHNOLOGIES MEET ALL REQUIREMENTS")
        print("   Consider parameter adjustments or hybrid approaches")

def main():
    """
    Complete anti-stiction metasurface coating demonstration
    """
    print("CASIMIR ANTI-STICTION METASURFACE COATING DEMONSTRATION")
    print("=" * 80)
    print("Comprehensive analysis of mathematical formulations and optimization")
    print("Target Specifications:")
    print("  • Static pull-in gap: ≥5 nm")
    print("  • Work of adhesion: ≤10 mJ/m²")
    print("  • Enhancement factor: ≥100×")
    print("  • Surface roughness: <0.2 nm RMS")
    
    try:
        # Run all demonstrations
        demo_metamaterial_enhancement()
        demo_casimir_force_calculation()
        demo_pull_in_gap_optimization()
        demo_sam_work_of_adhesion()
        demo_coating_optimization()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("All analyses completed successfully!")
        print("Generated files:")
        print("  • metamaterial_enhancement_analysis.png")
        print("  • casimir_force_analysis.png")
        print("  • pull_in_gap_optimization.png")
        print("  • sam_work_of_adhesion_analysis.png")
        print("  • anti_stiction_*_detailed_report.txt")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("  pip install numpy matplotlib")

if __name__ == "__main__":
    main()
