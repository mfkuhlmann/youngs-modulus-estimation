import matplotlib.pyplot as plt
from pathlib import Path
from model.VGG_LSTM.modulus_model import ModulusModel as VGG_LSTM
from model.Res_TF.modulus_model import ModulusModel as Res_TF
from model.Top10NN.modulus_model import ModulusModel as Top10NN
from dataset.k_fold_youngs_data_module import KFoldYoungsDataModule
from collections import Counter
import torch
import numpy as np
import tqdm
import gc
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Configure global font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.autolayout'] = True  # Enable better layout

# Model class mapping
model_class_dict = {
    "vgg_lstm": VGG_LSTM,
    "res_tf": Res_TF,
    "top10nn": Top10NN,
}

# Model display name mapping
model_name_map = {
    "vgg": "VGG-LSTM",
    "res": "Res-Tf",
    "top10nn": "Top10NN",
}

# Model checkpoint paths
model_checkpoints = [
    Path("<add your path here>").resolve(), 
    ... # Add other model paths here
    ]

# Material category color mapping
material_categories = {
    'Foam': '#e41a1c',     
    'Rubber': '#377eb8',   
    'Food': '#984ea3',     
    'Plastic': '#ff7f00',  
    'Wood': '#8c564b',     
    'Glass': '#e879bf',    
    'Metal': '#4daf4a',    
    'Ceramic': '#cccc00'   
}

# Create figure and grid layout with more space between plots
fig = plt.figure(figsize=(24, 16))
gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.2], hspace=0.3, wspace=0.3)

# Create subplot axes
axes = []
for i in range(2):
    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)

# Create legend area
legend_ax = fig.add_subplot(gs[:, 3])
legend_ax.axis('off')

# Process each model checkpoint
for model_idx, model_folder in enumerate(model_checkpoints):
    ax = axes[model_idx]
    
    # Initialize data containers
    all_predicted = []
    all_ground_truth = []
    all_herzian = []
    all_counter = []
    all_materials = []
    all_object_names = []
    
    # Determine model type and variant
    model_name = model_folder.stem.split("_")[0]
    is_all = 'est' in model_folder.stem.lower()
    model_variant = "ALL" if is_all else "Image"
    
    print(f"Processing model {model_idx+1}/6: {model_name} {model_variant}")
    
    # Load appropriate model
    if model_name == "top10nn":
        model = Top10NN.load_from_checkpoint(model_folder)
    elif model_name == "vgg":
        model = VGG_LSTM.load_from_checkpoint(model_folder)
    elif model_name == "res":
        model = Res_TF.load_from_checkpoint(model_folder)

    # Configure model
    model.eval()
    model.cuda(0)
    model.hparams.use_transformations = False
    model_config = model.config
    fold = int(model_folder.parent.stem.split("_")[-1])

    # Initialize data module
    data_module = KFoldYoungsDataModule(
        data_dir=model_config['data_dir'],
        training_data_folder=model_config['training_data_folder'],
        worker=5,
        image_style=model_config['img_style'],
        sample_type=model_config['sample_type'],
        use_estimations=model_config['use_estimations'],
        use_force=model_config['use_force'],
        use_width=model_config['use_width'],
        use_width_transforms=model_config['use_width_transforms'],
        use_markers=model_config['use_markers'],
        remove_paper=model_config['remove_paper'],
        overwrite_file=False,
        batch_size=1,
        val_on_seen_objects=model_config["val_on_seen_objects"],
        use_log_normalization=model_config['use_log_normalization'],
        exclude=model_config['exclude'],
    )
    data_module.make_object_attributes()
    data_module.prepare_data()
    data_module.setup(fold=fold)

    # Track prediction quality
    bad_objects = []
    good_objects = []
    
    # Process test data
    material_mapping = {}
    for batch in tqdm.tqdm(data_module.test_dataloader()):
        x_frames, x_forces, x_widths, x_estimations, y, object_name = batch
        outputs = model.forward(x_frames.cuda(), x_forces.cuda(), x_widths.cuda(), x_estimations.cuda(), x_frames.shape[0])
        outputs = outputs.squeeze()

        # Determine prediction quality
        if torch.abs(torch.log10(model.unnormalization_function(outputs.cpu())) - torch.log10(model.unnormalization_function(y.cpu()))) >= 1:
            bad_objects.append(object_name)
        else:
            good_objects.append(object_name)
        
        # Store results
        all_predicted.append(model.unnormalization_function(outputs.cpu()).detach().numpy())
        all_ground_truth.append(model.unnormalization_function(y.cpu()).detach().numpy())
        all_herzian.append(x_estimations.cpu().detach().numpy())
        
        # Categorize material
        material = data_module.object_to_material[object_name[0]]
        if 'rubber' in material.lower() or 'silicone' in material.lower():
            category = 'Rubber'
        elif 'metal' in material.lower() or 'steel' in material.lower() or 'aluminum' in material.lower() or 'bronze' in material.lower() or 'iron' in material.lower():
            category = 'Metal'
        elif 'wood' in material.lower():
            category = 'Wood'
        elif 'plastic' in material.lower() or 'pla' in material.lower() or 'abs' in material.lower():
            category = 'Plastic'
        elif 'glass' in material.lower():
            category = 'Glass'
        elif 'ceramic' in material.lower() or 'porcelain' in material.lower():
            category = 'Ceramic'
        elif 'foam' in material.lower() or 'sponge' in material.lower():
            category = 'Foam'
        elif 'food' in material.lower() or 'fruit' in material.lower() or 'vegetable' in material.lower():
            category = 'Food'
        else:
            for key in material_categories.keys():
                if key.lower() in material.lower():
                    category = key
                    break
            else:
                category = 'Plastic'
                
        all_materials.append(category)
        all_object_names.append(object_name[0])
    
    # Process object performance statistics
    all_counter.append([Counter(bad_objects), Counter(good_objects)])
    for key in all_counter[-1][0].keys():
        print(f"{key[0]}: {all_counter[-1][0][key]}/{all_counter[-1][1][key]} {data_module.object_to_material[key[0]]} {data_module.object_to_shape[key[0]]}")

    # Convert lists to arrays for calculations
    all_predicted = np.array(all_predicted).flatten()
    all_ground_truth = np.array(all_ground_truth).flatten()
    all_herzian = np.concatenate(all_herzian)
    
    # Get normalization values
    x_min = model.normalization_values['min_modulus']
    x_max = model.normalization_values['max_modulus']
    
    # Log normalization function
    def log_normalize(x, x_min, x_max):
        return (np.log10(x) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
    
    # Calculate normalized values
    norm_pred = log_normalize(all_predicted, x_min, x_max)
    norm_true = log_normalize(all_ground_truth, x_min, x_max)
    
    # Calculate N-MSE
    nmse = np.mean((norm_pred - norm_true) ** 2)
    print(f"NMSE: {nmse:.4f}")
    
    # Calculate boundary factor
    sqrt_nmse = np.sqrt(nmse)
    log_range = np.log10(x_max) - np.log10(x_min)
    log_factor = sqrt_nmse * log_range
    nmse_factor = 10 ** log_factor
    print(f"NMSE factor for boundaries: {nmse_factor:.4f}")
    
    # Configure axis appearance
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', length=0)
    
    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Force specific decades to show on both axes (more explicit method)
    decades = range(3, 13)  # 10^3 to 10^12
    
    # Explicitly set ticks and format them manually
    x_ticks = [10**i for i in decades]
    ax.set_xticks(x_ticks)
    
    y_ticks = [10**i for i in decades]
    ax.set_yticks(y_ticks)
    
    # Create lists of tick labels with explicit formatting
    x_tick_labels = [r'$10^{%d}$' % i for i in decades]
    y_tick_labels = [r'$10^{%d}$' % i for i in decades]
    
    # Apply the labels
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    
    # Make sure the ticks are visible by adjusting rotation and alignment
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Set explicit view limits
    ax.set_xlim(1e3, 1e12)
    ax.set_ylim(1e3, 1e12)
    
    # Turn off minor ticks completely
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    
    # Enhance plot appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    # Add reference lines
    ax.plot([1e3, 1e12], [1e3, 1e12], 'r--', lw=2.0)
    ax.plot([1e3, 1e12], [1e3 * nmse_factor, 1e12 * nmse_factor], 'k--', lw=1.5)
    ax.plot([1e3, 1e12], [1e3 / nmse_factor, 1e12 / nmse_factor], 'k--', lw=1.5)
    
    # Fill area between NMSE boundaries
    ax.fill_between([1e3, 1e12], 
                    [1e3 * nmse_factor, 1e12 * nmse_factor], 
                    [1e3 / nmse_factor, 1e12 / nmse_factor], 
                    color='lightgray', alpha=0.2)
    
    # Determine points within boundaries
    within_boundaries = np.logical_and(
        all_predicted <= all_ground_truth * nmse_factor,
        all_predicted >= all_ground_truth / nmse_factor
    )
    
    # Calculate percentage of points within boundaries
    within_count = np.sum(within_boundaries)
    total_count = len(all_predicted)
    within_percentage = (within_count / total_count) * 100
    print(f"Points within N-MSE boundaries: {within_count}/{total_count} ({within_percentage:.1f}%)")
    
    # Reshape arrays for plotting
    all_predicted = all_predicted.reshape(-1, 1)
    all_ground_truth = all_ground_truth.reshape(-1, 1)
    
    # Plot points by material category
    for category in material_categories:
        category_indices = [i for i, mat in enumerate(all_materials) if mat == category]
        if category_indices:
            # Points within boundaries (full opacity)
            within_indices = [i for i in category_indices if within_boundaries[i]]
            if within_indices:
                ax.scatter(
                    all_ground_truth[within_indices], 
                    all_predicted[within_indices], 
                    color=material_categories[category],
                    alpha=0.7, 
                    s=50,
                    label=""
                )
            
            # Points outside boundaries (reduced opacity)
            outside_indices = [i for i in category_indices if not within_boundaries[i]]
            if outside_indices:
                base_color = mcolors.to_rgba(material_categories[category])
                light_color = (base_color[0], base_color[1], base_color[2], 0.3)
                
                ax.scatter(
                    all_ground_truth[outside_indices], 
                    all_predicted[outside_indices], 
                    color=light_color,
                    s=50,
                    label=""
                )
    
    # Add Gelsight reference line
    gelsight_value = 275000
    ax.axvline(x=gelsight_value, color='blue', linestyle='--', lw=1.5, alpha=0.7)
    
    # Add Gelsight label
    x_pos = gelsight_value * 0.10
    y_pos = 1e11
    ax.text(x_pos, y_pos, '0.275 MPa', color='blue', 
            ha='center', va='center', fontsize=14)
    
    # Add x-axis labels to bottom row only
    if model_idx >= 3:
        ax.set_xlabel('Real Young\'s modulus (Pa)', fontsize=18)
    
    # Add y-axis labels to leftmost column only
    if model_idx % 3 == 0:
        ax.set_ylabel('Predicted Young\'s modulus (Pa)', fontsize=18)
    
    # Calculate R² for reference
    correlation_matrix = np.corrcoef(np.log10(all_ground_truth.flatten()), np.log10(all_predicted.flatten()))
    r_squared = correlation_matrix[0, 1] ** 2
    
    # Set plot title
    ax.set_title(f'{model_name_map[model_name]} {model_variant}\nN-MSE = {nmse:.4f}', fontsize=20)
    
    # Clean up memory
    del data_module
    gc.collect()

# Create legend elements
legend_elements = []

# Add material category markers
for category in material_categories:
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=material_categories[category],
               markersize=14, label=category)
    )

# Add ideal fit line
legend_elements.append(
    Line2D([0], [0], color='r', linestyle='--', lw=2.5, label='Ideal Fit (y=x)')
)

# Create legend
legend = legend_ax.legend(
    title="Materials",
    handles=legend_elements,
    loc='center',
    frameon=True,
    framealpha=0.9,
    fontsize=16
)
legend.get_title().set_fontsize(20)

# Adjust layout with more padding
plt.tight_layout(pad=2.0)

# Save output files at higher resolution
plt.savefig("six_models_comparison.pdf", dpi=400, bbox_inches='tight')
plt.savefig("six_models_comparison.png", dpi=400, bbox_inches='tight')

print("Figure saved successfully.")