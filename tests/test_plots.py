"""
Test script for FPTHandler plotting functions using real FASTPT data
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from fastpt import FASTPT
from fastpt.FPTHandler import FPTHandler

def test_plotting_features():
    """Comprehensive test of FPTHandler plotting features with actual FASTPT data"""
    # Create output directory for plots
    output_dir = "plot_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the path to the example data file
    # First try the directory of this test file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(os.path.dirname(current_dir), 'examples')
    data_path = os.path.join(example_dir, 'Pk_test.dat')
    
    # If not found, try a relative path from current directory
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(current_dir), 'FAST-PT', 'examples', 'Pk_test.dat')
    
    # If still not found, use a default test path
    if not os.path.exists(data_path):
        data_path = 'Pk_test.dat'
        
    print(f"Loading data from: {data_path}")
    
    # Load the power spectrum data
    try:
        data = np.loadtxt(data_path)
        k = data[:, 0]
        P_linear = data[:, 1]
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Falling back to synthetic data for testing...")
        # Generate synthetic data if real data can't be loaded
        k = np.logspace(-3, 1, 200)
        P_linear = k**(-1.5) * 1000  # Simple power law
    
    # Initialize FASTPT
    print("Initializing FASTPT...")
    n_pad = int(0.5 * len(k))
    fpt = FASTPT(k, to_do=['all'], low_extrap=-5, high_extrap=3, n_pad=n_pad)
    
    # Initialize FPTHandler
    print("Initializing FPTHandler...")
    handler = FPTHandler(fpt, P=P_linear, C_window=0.75)
    
    # Compute actual FASTPT results
    print("Computing FASTPT results...")
    
    # 1-loop Standard Perturbation Theory terms
    P_spt = fpt.one_loop_dd(P_linear, C_window=0.75)
    
    # Intrinsic alignment terms
    P_IA_tt = fpt.IA_tt(P_linear, C_window=0.75)
    P_IA_ta = fpt.IA_ta(P_linear, C_window=0.75)
    P_IA_mix = fpt.IA_mix(P_linear, C_window=0.75)
    
    # RSD terms with bias=1.0
    P_RSD = fpt.RSD_components(P_linear, 1.0, C_window=0.75)
    
    # Other components
    P_kPol = fpt.kPol(P_linear, C_window=0.75)
    P_OV = fpt.OV(P_linear, C_window=0.75)
    
    print("\n===== TESTING BASIC PLOTTING =====")
    
    # Test 1: Basic plotting from data dictionary
    print("\nTest 1: Basic plotting - linear and 1-loop SPT")
    data_dict = {
        'Linear': P_linear,
        'P_22+P_13': P_spt[0]
    }
    fig = handler.plot(
        data=data_dict, 
        title="Linear P(k) and 1-loop Correction",
        save_path=os.path.join(output_dir, "test1_basic_spt.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 2: Plotting from function results
    print("\nTest 2: Plotting IA terms")
    # Plot E and B modes from tidal torque
    fig = handler.plot(
        data={
            'IA E-mode (TT)': P_IA_tt[0],
            'IA B-mode (TT)': P_IA_tt[1]
        },
        title="IA Tidal Torque E/B Modes",
        save_path=os.path.join(output_dir, "test2_IA_tt.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 3: Custom styling using style dictionaries
    print("\nTest 3: Custom styling for RSD terms")
    # Based on FASTPT implementation, we know P_RSD[0] access is correct and contains multiple rows
    rsd_data = {
        'RSD Component 1': P_RSD[0],
        'RSD Component 2': P_RSD[1],
        'RSD Component 3': P_RSD[2]
    }
    fig = handler.plot(
        data=rsd_data,
        style=[
            {'color': 'red', 'linestyle': '-', 'linewidth': 2.5},
            {'color': 'blue', 'linestyle': '--', 'linewidth': 2.0},
            {'color': 'green', 'linestyle': '-.', 'linewidth': 1.5}
        ],
        title="RSD Components with Custom Styling",
        save_path=os.path.join(output_dir, "test3_rsd_styled.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 4: Custom labels and scaling for IA mix terms
    print("\nTest 4: Custom labels and scaling for IA mix terms")
    ia_mix_data = {
        'IA_mix[0]': P_IA_mix[0],
        'IA_mix[1]': P_IA_mix[1]
    }
    
    # Some terms may be small, so scale them for better visibility
    label_map = {
        'IA_mix[0]': 'IA mix term 1',
        'IA_mix[1]': 'IA mix term 2'
    }
    # Scale up these terms to make them more visible
    scale_factors = {
        'IA mix term 1': 10.0,
        'IA mix term 2': 10.0
    }
    
    fig = handler.plot(
        data=ia_mix_data,
        label_map=label_map,
        scale_factors=scale_factors,
        title="IA Mix Terms (scaled 10x)",
        save_path=os.path.join(output_dir, "test4_IA_mix_scaled.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 5: Handling positive and negative values
    print("\nTest 5: Negative value handling with P_13")
    # P_spt[0] is P22+P13, P_spt[1] is the Ps term
    # In FAST-PT, P22 and P13 aren't directly accessible, so we'll use synthetic data
    # that mimics the behavior of having negative values
    p13_p22_data = {
        'P_22': P_linear * 0.2,  # Synthetic positive term
        'P_13': P_linear * -0.1 * np.sin(5*np.log(k)),  # Synthetic term with negative values
        'P_22+P_13': P_spt[0]  # Actual P22+P13 from FAST-PT
    }
    
    fig = handler.plot(
        data=p13_p22_data,
        title="P_22 and P_13 Components",
        save_path=os.path.join(output_dir, "test5_P22_P13.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 6: Custom axis limits for kPol terms
    print("\nTest 6: kPol terms with custom axis limits")
    # kPol returns a tuple of three arrays
    kpol_data = {
        'kPol[0]': P_kPol[0],
        'kPol[1]': P_kPol[1],
        'kPol[2]': P_kPol[2]
    }
    
    fig = handler.plot(
        data=kpol_data,
        xlim=(0.01, 1.0),
        ylim=(None, None),  # Auto y limits
        title="k-dependent Polarization Terms",
        save_path=os.path.join(output_dir, "test6_kPol_limits.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 7: Using markers for OV terms
    print("\nTest 7: OV terms with markers")
    # P_OV is a single array
    ov_data = {
        'OV': P_OV
    }
    
    fig = handler.plot(
        data=ov_data,
        style=[
            {'marker': 'o', 'markersize': 4, 'markevery': 10}
        ],
        title="Optical Depth Components",
        save_path=os.path.join(output_dir, "test7_OV_markers.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 8: Log/linear scale options with IA tidal alignment
    print("\nTest 8: IA tidal alignment with x-log, y-linear scale")
    ia_ta_data = {
        'IA_ta[0]': P_IA_ta[0],
        'IA_ta[1]': P_IA_ta[1]
    }
    
    fig = handler.plot(
        data=ia_ta_data,
        log_scale=(True, False),  # x-log, y-linear
        title="IA Tidal Alignment (x-log, y-linear)",
        save_path=os.path.join(output_dir, "test8_IA_ta_scale.png"),
        return_fig=True,
        show=False
    )
    plt.close(fig)
    
    # Test 9: Custom axes for comparing P_linear and full 1-loop result
    print("\nTest 9: Full 1-loop power spectrum with custom axes")
    fig, ax = plt.subplots(figsize=(10, 7))
    # Add a horizontal line at P=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add a vertical line at non-linear scale (example)
    k_nl = 0.1
    ax.axvline(x=k_nl, color='gray', linestyle='--', alpha=0.5)
    ax.text(k_nl*1.1, ax.get_ylim()[1]*0.9, r'$k_{NL}$', fontsize=12)
    
    # Full 1-loop power spectrum: P_linear + (P_22 + P_13)
    P_1loop_full = P_linear + P_spt[0]
    
    # Use the custom axes for plotting
    handler.plot(
        data={
            'Linear': P_linear,
            '1-loop SPT': P_1loop_full
        },
        ax=ax,
        title="Linear vs. Full 1-loop Power Spectrum",
        save_path=os.path.join(output_dir, "test9_full_1loop.png"),
        show=False
    )
    plt.close(fig)
    
    print("\n===== TESTING COMPARISON PLOTS =====")
    
    # Test 10: Basic comparison plot with bias models
    print("\nTest 10: Compare different bias models")
    # Create simple bias models by scaling linear P(k)
    b1 = 1.5  # Example bias parameter
    b2 = 2.0
    b3 = 2.5
    
    comparison_data = {
        'Linear': P_linear,
        f'Biased (b={b1})': b1**2 * P_linear,
        f'Biased (b={b2})': b2**2 * P_linear,
        f'Biased (b={b3})': b3**2 * P_linear
    }
    fig = handler.plot_comparison(
        comparison_data,
        title="Comparison of Different Bias Models",
        save_path=os.path.join(output_dir, "test10_bias_compare.png"),
        show=False
    )
    plt.close(fig)
    
    # Test 11: Comparison plot with ratio panel
    print("\nTest 11: Compare linear and 1-loop with ratio panel")
    ratio_data = {
        'Linear': P_linear,
        '1-loop': P_1loop_full,
        'Model 2': 1.05 * P_linear,
        'Model 3': 0.95 * P_linear
    }
    
    fig = handler.plot_comparison(
        ratio_data,
        ratio=True,
        ratio_baseline='Linear',
        title="1-loop SPT Components vs. Linear",
        save_path=os.path.join(output_dir, "test11_1loop_ratio.png"),
        show=False
    )
    plt.close(fig)
    
    # Test 12: Comprehensive styling with comparison of RSD multipoles
    print("\nTest 12: RSD multipoles with custom styling")
    # RSD multipoles are in P_RSD[1]
    rsd_multipoles = {
        'Monopole': P_RSD[1][0],
        'Quadrupole': P_RSD[1][1],
        'Hexadecapole': P_RSD[1][2]
    }
    
    fig = handler.plot_comparison(
        rsd_multipoles,
        ratio=True,
        ratio_baseline='Monopole',
        colors=['black', 'red', 'blue'],
        style=[
            {'linestyle': '-', 'linewidth': 2},
            {'linestyle': '--', 'linewidth': 2},
            {'linestyle': '-.', 'linewidth': 2}
        ],
        title="RSD Multipoles",
        save_path=os.path.join(output_dir, "test12_rsd_multipoles.png"),
        show=False
    )
    plt.close(fig)
    
    # Test 13: Comprehensive comparison of power spectrum components
    print("\nTest 13: Comprehensive comparison of power spectrum components")
    complex_data = {
        'Linear': P_linear,
        '1-loop SPT': P_1loop_full,
        'IA E-mode (TT)': P_IA_tt[0],
        'IA B-mode (TT)': P_IA_tt[1],
        'RSD Monopole': P_RSD[1][0]
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    # Add a text annotation to the plot
    ax.text(0.05, 0.95, 'Power Spectrum Components', transform=ax.transAxes, 
            fontsize=12, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
    
    # Create a span to highlight BAO region (example)
    ax.axvspan(0.05, 0.3, alpha=0.1, color='green')
    
    handler.plot(
        data=complex_data,
        ax=ax,
        colors=['black', 'red', 'blue', 'green', 'purple'],
        style=[
            {'linestyle': '-', 'linewidth': 2},
            {'linestyle': '--', 'linewidth': 2},
            {'linestyle': '-.', 'marker': 'o', 'markevery': 15},
            {'linestyle': ':', 'marker': 's', 'markevery': 15},
            {'linestyle': '-', 'linewidth': 1.5}
        ],
        title="Power Spectrum Components",
        save_path=os.path.join(output_dir, "test13_comprehensive.png"),
        show=False
    )
    plt.close(fig)
    
    print("\nAll tests completed! Output images saved to:", output_dir)
    print("Inspect the generated plots to verify all features work correctly.")
    
if __name__ == "__main__":
    test_plotting_features()