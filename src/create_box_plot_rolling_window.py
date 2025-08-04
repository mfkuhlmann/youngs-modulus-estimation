from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.ticker as ticker



def create_final_boxplot(path_to_results):
    """Create the final boxplot with all requested changes"""
    
    # Set plot style with even larger fonts
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 24  # Increase base font size further
    plt.rcParams['axes.titlesize'] = 30  # Increase title font size
    plt.rcParams['axes.labelsize'] = 28  # Increase axis label font size
    plt.rcParams['xtick.labelsize'] = 24  # Increase x-tick label font size
    plt.rcParams['ytick.labelsize'] = 24  # Increase y-tick label font size
    plt.rcParams['legend.fontsize'] = 24  # Increase legend font size
    plt.rcParams['legend.title_fontsize'] = 26  # Increase legend title font size
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Set order of categories
    #shape_order = ['Sphere', 'Cylinder', 'Rectangular', 'Irregular']
    
    # Order: VGG Image, VGG ALL, Res Image, Res ALL
    model_input_order = ["vgg_lstm", "res_tf"] # ['VGG-LSTM','Res-Tf']
    key_to_name = {
        "vgg_lstm": "VGG-LSTM Image",
        "res_tf": "Res-Tf Image"
    }
    # Define colors for each model/input combination
    colors = {
        'vgg_lstm': '#5ec962',   # Orange
        #'VGG-LSTM ALL': '#35b779',     # Blue
        'res_tf': '#3b528b',     # Red
        #'Res-TF ALL': '#440154',       # Green
    }
    
    # Calculate positions for each box with wider spacing
    positions = {}
    box_width = 0.15  # Make boxes thinner
    gap = 0.1  # Gap between boxes within a group
    group_gap = 0.8  # Gap between shape groups

    #Load data
    res = {}
    for model_folder in path_to_results.iterdir():
        if model_folder.stem not in model_input_order:
            continue
        res[model_folder.stem] = {}
        for rolling_window in range(7):
            res[model_folder.stem][rolling_window] = []
            for fold in range(10):
                path_to_result_numbers = model_folder / f"{fold}" / f"{model_folder.stem.split('_')[-1]}_random_seen_both_log_normalize_removed_paper_rolling_window_{rolling_window}_undersampled_no_cross_validation_random_state_0_random.pkl"
                with path_to_result_numbers.open("rb") as f:
                    result_numbers = pickle.load(f)
                res[model_folder.stem][rolling_window].append(result_numbers["n_mse_loss"])
    print(res)
    # Plot each boxplot manually
    boxplots = []
    model_offset = {
        "res_tf": 0.15,
        "vgg_lstm": -0.15
    }
    for model_input in model_input_order:
            # Filter data for this combination
            x= [res[model_input][rolling_window] for rolling_window in range(7)]
            # Create the boxplot without fliers (outliers)
            bp = ax.boxplot(
                x,
                positions=[k + model_offset[model_input] for k in range(7)],
                widths=box_width,
                patch_artist=True,
                showfliers=False,  # Remove outliers
                medianprops={'color': 'black', 'linewidth': 2.0}  # Make median line even thicker
            )
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor(colors[model_input])
                patch.set_alpha(0.7)
                patch.set_linewidth(2.0)  # Make box borders even thicker
                
            # Make whiskers and caps thicker
            for whisker in bp['whiskers']:
                whisker.set_linewidth(2.0)
                
            for cap in bp['caps']:
                cap.set_linewidth(2.0)
                
            boxplots.append((model_input, bp))
    
    # Explicitly set y-axis ticks
    y_ticks = [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
    ax.set_yticks(y_ticks)
    
    # Format y-axis tick labels
    y_tick_labels = ['0.000', '0.001', '0.002', '0.003', '0.004', '0.005', '0.006', '0.007', '0.008', '0.009', '0.010']
    ax.set_yticklabels(y_tick_labels)
    
    # Set y-axis limits
    ax.set_ylim(0.0000, 0.010)
    
    # Set x-axis ticks at the center of each shape group
    #group_centers = []
    #for i, shape in enumerate(shape_order):
    #    start = positions[(shape, model_input_order[0])]
    #    end = positions[(shape, model_input_order[-1])]
    #    center = (start + end) / 2 + box_width/2
    #    group_centers.append(center)
    
    ax.set_xticks(list(range(7)))
    x_labels = [f"1e{3 + i} - 1e{6 + i}" for i in range(7)]
    ax.set_xticklabels(x_labels)
    
    # Add labels and title
    ax.set_xlabel('Window Range', fontsize=40, labelpad=20)  # Increase labelpad for more space
    ax.set_ylabel('N-MSE', fontsize=40, labelpad=20)
    # ax.set_title('N-MSE Comparison by different Shapes', fontsize=35, pad=25)  # Add more padding
    
    # Make tick marks face inward with appropriate padding to avoid overlap
    # For y-axis: Make ticks point inward but keep labels outside
    ax.tick_params(axis='y', direction='in', pad=15, length=10, width=2.0)
    
    # For x-axis: Make ticks point inward but keep labels below
    ax.tick_params(axis='x', direction='in', pad=15, length=10, width=2.0)
    
    # Add ticks on the top and right sides
    ax.tick_params(axis='x', top=True, direction='in', pad=15, length=10, width=2.0, labelsize=30)
    ax.tick_params(axis='y', right=True, direction='in', pad=15, length=10, width=2.0, labelsize=30)
    
    # Add vertical dashed lines to separate shape groups
    for i in range(6):
        #prev_end = positions[(shape_order[i-1], model_input_order[-1])] + box_width
        #next_start = positions[(shape_order[i], model_input_order[0])]
        separator_pos = i + 0.5
        ax.axvline(separator_pos, color='gray', linestyle='--', alpha=0.7, linewidth=2.0)
    
    # Remove all grid lines
    ax.grid(False)
    
    # Create a custom legend with larger elements
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[mi], alpha=0.7, 
                      linewidth=2.0, edgecolor='black', label=key_to_name[mi]) 
        for mi in model_input_order
    ]
    ax.legend(
        handles=legend_elements, 
        title='Model & Input', 
        loc='best', 
        fontsize=26,
        title_fontsize=26,
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    
    # Make the plot spines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    # Improve layout with more padding
    plt.tight_layout(pad=2.0)
    
    # Save the figure
    plt.savefig('../figures/nmse_rolling_window.png', dpi=400, bbox_inches='tight')
    plt.savefig('../figures/nmse_rolling_window.pdf', dpi=400, bbox_inches='tight')
    plt.savefig('../figures/nmse_rolling_window.svg', dpi=400, bbox_inches='tight')
    
    return plt

if __name__ == "__main__":
    path_to_results = Path("../data/results_rolling_window_undersampled_2").resolve()
    

    # Create the final boxplot
    create_final_boxplot(path_to_results)
    