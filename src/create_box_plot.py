import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.ticker as ticker

def parse_data(data_text):
    """Parse the NMSE data from the text file using a direct approach"""
    
    # Initialize a list to store all the data rows
    data_rows = []
    
    # Split the text into lines
    lines = data_text.strip().split('\n')
    
    current_experiment = None
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # If line starts with 'lstm' or 'tf', it's an experiment name
        if line.startswith('lstm_') or line.startswith('tf_'):
            current_experiment = line.strip()
            continue
            
        # Skip the header line that starts with "Model Name"
        if line.startswith('Model Name'):
            continue
            
        # If line starts with 'vgg_lstm' or 'res_tf', it contains the data
        if line.startswith('vgg_lstm') or line.startswith('res_tf'):
            # No current experiment, skip
            if not current_experiment:
                continue
                
            # Extract model name
            model_name = 'VGG-LSTM' if line.startswith('vgg_lstm') else 'Res-TF'
            
            # Extract shape from experiment name
            shape_match = re.search(r'(Sphere|Cylinder|Rectangular|Irregular)', current_experiment)
            shape = shape_match.group(1) if shape_match else "Unknown"
            
            # Determine input type (ALL with est_force_width or Image with just both)
            input_type = "ALL" if "est_force_width" in current_experiment else "Image"
            
            # Split the line by spaces and extract the 10 fold values
            parts = line.split()
            # Get numerical values - start from index 1 to skip the model name
            values = []
            for val in parts[1:11]:  # Take the first 10 values
                # Replace comma with dot for decimal
                val_fixed = val.replace(',', '.')
                try:
                    values.append(float(val_fixed))
                except ValueError:
                    print(f"Error converting value: {val}")
                    values.append(np.nan)
            
            # For each value, add a row to our data
            for fold, nmse in enumerate(values):
                data_rows.append({
                    'Model': model_name,
                    'InputType': input_type,
                    'Shape': shape,
                    'NMSE': nmse,
                    'Fold': fold
                })
    
    # Create DataFrame from the collected data
    df = pd.DataFrame(data_rows)
    
    # Print some debug information
    print(f"Parsed data shape: {df.shape}")
    if not df.empty:
        print(f"Models: {df['Model'].unique().tolist()}")
        print(f"Input Types: {df['InputType'].unique().tolist()}")
        print(f"Shapes: {df['Shape'].unique().tolist()}")
        
    return df

def create_final_boxplot(df):
    """Create the final boxplot with all requested changes"""
    
    # Create a new column for model+input combination
    df['ModelInput'] = df['Model'] + ' ' + df['InputType']
    
    # Set plot style with even larger fonts
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 24  # Increase base font size further
    plt.rcParams['axes.titlesize'] = 30  # Increase title font size
    plt.rcParams['axes.labelsize'] = 28  # Increase axis label font size
    plt.rcParams['xtick.labelsize'] = 28  # Increase x-tick label font size
    plt.rcParams['ytick.labelsize'] = 28  # Increase y-tick label font size
    plt.rcParams['legend.fontsize'] = 24  # Increase legend font size
    plt.rcParams['legend.title_fontsize'] = 26  # Increase legend title font size
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Set order of categories
    shape_order = ['Sphere', 'Cylinder', 'Rectangular', 'Irregular']
    
    # Order: VGG Image, VGG ALL, Res Image, Res ALL
    model_input_order = ['VGG-LSTM Image', 'VGG-LSTM ALL', 'Res-TF Image', 'Res-TF ALL']
    
    # Define colors for each model/input combination
    colors = {
        'VGG-LSTM Image': '#fde725',   # Orange
        'VGG-LSTM ALL': '#35b779',     # Blue
        'Res-TF Image': '#31688e',     # Red
        'Res-TF ALL': '#440154',       # Green
    }
    
    # Calculate positions for each box with wider spacing
    positions = {}
    box_width = 0.15  # Make boxes thinner
    gap = 0.1  # Gap between boxes within a group
    group_gap = 0.8  # Gap between shape groups
    
    for i, shape in enumerate(shape_order):
        base_pos = i * (4 * (box_width + gap) + group_gap)
        for j, model_input in enumerate(model_input_order):
            positions[(shape, model_input)] = base_pos + j * (box_width + gap)
    
    # Plot each boxplot manually
    boxplots = []
    for shape in shape_order:
        for model_input in model_input_order:
            # Filter data for this combination
            subset = df[(df['Shape'] == shape) & (df['ModelInput'] == model_input)]
            
            if subset.empty:
                continue
                
            # Create the boxplot without fliers (outliers)
            pos = positions[(shape, model_input)]
            bp = ax.boxplot(
                subset['NMSE'].values,
                positions=[pos],
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
    y_ticks = [0.000, 0.005, 0.010, 0.015, 0.020]
    ax.set_yticks(y_ticks)
    
    # Format y-axis tick labels
    y_tick_labels = ['0', '0.005', '0.010', '0.015', '0.020']
    ax.set_yticklabels(y_tick_labels)
    
    # Set y-axis limits
    ax.set_ylim(0.0000, 0.020)
    
    # Set x-axis ticks at the center of each shape group
    group_centers = []
    for i, shape in enumerate(shape_order):
        start = positions[(shape, model_input_order[0])]
        end = positions[(shape, model_input_order[-1])]
        center = (start + end) / 2 + box_width/2
        group_centers.append(center)
    
    ax.set_xticks(group_centers)
    ax.set_xticklabels(shape_order)
    
    # Add labels and title
    ax.set_xlabel('Shape', fontsize=32, labelpad=20)  # Increase labelpad for more space
    ax.set_ylabel('N-MSE', fontsize=32, labelpad=20)
    # ax.set_title('N-MSE Comparison by different Shapes', fontsize=35, pad=25)  # Add more padding
    
    # Make tick marks face inward with appropriate padding to avoid overlap
    # For y-axis: Make ticks point inward but keep labels outside
    ax.tick_params(axis='y', direction='in', pad=15, length=10, width=2.0)
    
    # For x-axis: Make ticks point inward but keep labels below
    ax.tick_params(axis='x', direction='in', pad=15, length=10, width=2.0)
    
    # Add ticks on the top and right sides
    ax.tick_params(axis='x', top=True, direction='in', pad=15, length=10, width=2.0)
    ax.tick_params(axis='y', right=True, direction='in', pad=15, length=10, width=2.0)
    
    # Add vertical dashed lines to separate shape groups
    for i in range(1, len(shape_order)):
        prev_end = positions[(shape_order[i-1], model_input_order[-1])] + box_width
        next_start = positions[(shape_order[i], model_input_order[0])]
        separator_pos = (prev_end + next_start) / 2
        ax.axvline(separator_pos, color='gray', linestyle='--', alpha=0.7, linewidth=2.0)
    
    # Remove all grid lines
    ax.grid(False)
    
    # Create a custom legend with larger elements
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[mi], alpha=0.7, 
                      linewidth=2.0, edgecolor='black', label=mi) 
        for mi in model_input_order
    ]
    ax.legend(
        handles=legend_elements, 
        title='Model & Input', 
        loc='upper right', 
        fontsize=23,
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
    plt.savefig('../figures/nmse_final_boxplot.png', dpi=400, bbox_inches='tight')
    plt.savefig('../figures/nmse_final_boxplot.pdf', dpi=400, bbox_inches='tight')
    
    return plt

if __name__ == "__main__":
    # Read the data from a file
    # This uses a manual custom file. It will be updated at a later time to read actual checkpoints
    with open('<add your path here>', 'r') as file:
        data_text = file.read()
    
    # Parse the data
    df = parse_data(data_text)
    
    if df.empty:
        print("Error: No data was parsed from the file.")
    else:
        # Create the final boxplot
        create_final_boxplot(df)
        
        # Print info about the data for verification
        print("\nSummary statistics by model, input type, and shape:")
        summary = df.groupby(['Model', 'InputType', 'Shape'])['NMSE'].describe()
        print(summary)