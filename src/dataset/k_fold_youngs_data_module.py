from collections import Counter
import csv
import os
from pathlib import Path
import pickle
import lightning
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, train_test_split, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
import torchvision
import torchvision.transforms.v2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import cv2

from dataset.offloaded_dataset import CustomDataset
from dataset.utils import list_files

class YoungsDataset(Dataset):
    def __init__(self, data, apply_augmentations=False, augmentation_instructions=None, use_width=False) -> None:
        self.data = data
        self.apply_augmentations = apply_augmentations
        self.use_width = use_width
        self.p = 0.1
        self.random_transformer = torchvision.transforms.Compose([
            torchvision.transforms.v2.RandomHorizontalFlip(0.5),
            torchvision.transforms.v2.RandomVerticalFlip(0.5),
            torchvision.transforms.v2.ColorJitter(brightness=.4, contrast=0.4, saturation=0.5, hue=.3),
            torchvision.transforms.v2.GaussianNoise(mean=0, sigma=0.05882352941), # Always apply noise
        ])
    
    def __len__(self):
        return len(self.data["x_frames"])
    
    def width_transform(self, x_widths):
        noise_amplitude = min(
            1 - x_widths.max(),
            x_widths.min()
        )
        
        if torch.rand(1) > 0.5:
            x_widths += noise_amplitude * torch.rand(1)
        else:
            x_widths -= noise_amplitude * torch.rand(1)
        
        return x_widths
    
    def __getitem__(self, idx):
        x_frames = self.data["x_frames"][idx]
        x_widths = self.data["x_widths"][idx]
        if self.apply_augmentations:
            if torch.rand(1) > self.p: # Making sure that sometimes original data points are used
                x_frames = self.random_transformer(x_frames)
                if self.use_width:
                    x_widths = self.width_transform(x_widths)

                
        return x_frames, self.data["x_forces"][idx], x_widths, self.data["x_estimations"][idx], self.data["y_label"][idx], self.data["object_name"][idx]
    

class KFoldYoungsDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            data_dir,
            training_data_folder, 
            sample_type, 
            exclude, 
            exclude_shape=None, 
            compliance=None, 
            overwrite_file=False, 
            use_log_normalization=True, 
            worker=0, 
            batch_size=1, 
            val_on_seen_objects=False, 
            rubber_only=False, 
            use_estimations=False, 
            use_force=False, 
            use_width=False, 
            use_width_transforms=False, 
            labels_csv_name='dataset_objects_and_compliance.csv', 
            csv_modulus_column=14, 
            csv_shape_column=2, 
            csv_material_column=3, 
            image_style="RGB", 
            use_markers="both", 
            leave_out_unstratifiable_objects=False, 
            remove_paper = False, 
            stratify_with_magnitude=False, 
            n_splits=10, 
            use_cross_validation=True,
            same_test_set=True,
            balance_dataset=True,
            balance_position="before",
            balance_validation_set=True,
            balance_test_set=False,
            balance_threshold=500,
            balance_bucket=12,
            balance_undersample=False,
            full_dataset_undersample=False,
            full_dataset_undersample_amount=500,
            rolling_window=0,
            random_state = 42
        ) -> None:
        super().__init__()

        self.n_channels = 3
        # Normalize based on mean and std computed over the dataset
        if self.n_channels == 3:
            # Use the diff mean and std computed for our dataset
            self.image_normalization = torchvision.transforms.Normalize( \
                                            [0.49638007, 0.49770336, 0.49385751], \
                                            [0.04634926, 0.06181679, 0.07152624] \
                                        )
            
        self.data_dir = data_dir
        self.overwrite_file = overwrite_file
        self.worker = worker
        self.training_data_folder = training_data_folder
        self.batch_size = batch_size

        self.sample_type = sample_type
        self.stratify_with_magnitude = stratify_with_magnitude
        self.leave_out_unstratifiable_objects = leave_out_unstratifiable_objects

        self.val_on_seen_objects = val_on_seen_objects
        self.image_style = image_style
        self.use_markers = use_markers
        assert self.use_markers in ["both", "only_markers", "no_markers"]

        self.use_force = use_force
        self.use_width = use_width
        
        self.use_log_normalization = use_log_normalization
        self.use_estimations = use_estimations
        
        self.labels_csv_name = labels_csv_name
        self.csv_modulus_column = csv_modulus_column
        self.csv_shape_column = csv_shape_column
        self.csv_material_column = csv_material_column

        self.rubber_only = rubber_only
        self.exclude = exclude
        self.exclude_shape = exclude_shape
        self.compliance = compliance
        self.rolling_window = rolling_window
        self.remove_paper = remove_paper
        
        self.n_splits = n_splits
        self.use_cross_validation = use_cross_validation
        self.same_test_set = same_test_set

        assert (self.same_test_set != self.use_cross_validation) or (not self.same_test_set and not self.use_cross_validation) # both cannot be true
        
        self.random_state = random_state
        self.numpy_generator = np.random.default_rng(seed=self.random_state)

        self.should_balance = balance_dataset
        self.balance_position = balance_position
        assert self.balance_position in ["before", "after_all_folds", "after_isolated", "after_isolated_lower_memory"]
        self.balance_bucket = balance_bucket
        self.balance_threshold = balance_threshold
        self.balance_undersample = balance_undersample
        self.balance_augmentation_repetition = True
        self.balance_test_set = balance_test_set
        self.balance_validation_set = balance_validation_set

        self.full_dataset_undersample = full_dataset_undersample
        self.full_dataset_undersample_amount = full_dataset_undersample_amount

        self.config = {
            "img_style": image_style,
            "use_markers": use_markers,
            "training_data_folder": training_data_folder,
            "use_force": use_force,
            "use_width": use_width,
            "use_estimations": use_estimations,
            "use_width_transforms": False
        }

        if self.should_balance:
            self.config["use_width_transforms"] = False
        else:
            self.config["use_width_transforms"] = use_width_transforms

        # Create max values for scaling
        self.normalization_values = { # Based on acquired data maximums
            'min_modulus': 1e9,
            'max_modulus': 1e12,
            'min_estimate': 1e2,
            'max_estimate': 1e14,
            'max_depth': 7.0,
            'max_width': 0.08, # Measured from Franka Panda gripper
            'max_force': 60.0,
        }

        # Reduce prediction range if objects are all rubber
        if self.rubber_only:
            self.normalization_values['min_modulus'] = 1e5
            self.normalization_values['max_modulus'] = 1e8

        self.full_dataset_file = Path(f"{self.data_dir}/dataset_variants/{self.get_dataset_name()}").resolve()
        # self.full_dataset_file = Path("/home/malte.kuhlmann/youngs-modulus/data/dataset_variants/old/dataset_random_seen_width_transforms_log_normalize_removed_paper_random_state_42_random_dataset.pkl").resolve()
        print(self.full_dataset_file)

        if self.use_log_normalization:
            print("Using log normalization")
            self.normalization_function = self.log_normalize
            self.unnormalization_function = self.log_unnormalize
        else:
            print("Using standard normalization")
            self.normalization_function = self.normalize
            self.unnormalization_function = self.unnormalize
        #print(self.full_dataset_file.exists())
        #print(self.val_on_seen_objects)
        #quit()

    def get_dataset_name(self):
        name = "dataset"

        if self.rubber_only:
            name += "_rubber_only"

        name += f"_{self.sample_type}"

        if self.sample_type == "shape_material" and self.stratify_with_magnitude:
            name += "_magnitude"
        
        if self.val_on_seen_objects:
            name += "_seen"
        else:
            name += "_unseen"
        
        if self.use_estimations:
            name += "_est"
        
        if self.use_force:
            name += "_force"
        
        if self.use_width:
            name += "_width"
        
        if self.config["use_width_transforms"]:
            name += "_width_transforms"

        name += f"_{self.use_markers}"

        if self.use_log_normalization:
            name += "_log_normalize"
        else:
            name += "_normal_normalize"
        
        if self.exclude_shape is not None:
            name += f"_exclude_shape"
        
        if self.remove_paper:
            name += "_removed_paper"

        if self.compliance is not None:
            name += f"_{self.compliance}"
            if self.compliance == "rolling_window":
                name += f"_{self.rolling_window}"
        
        if self.full_dataset_undersample:
            name += "_undersampled"
        
        if self.n_splits != 10:
            name += f"_{self.n_splits}_splits"

        if self.should_balance:
            name += f"_balanced_{int(self.balance_threshold * 100)}"
        
        if not self.use_cross_validation:
            name += "_no_cross_validation"

        if self.same_test_set:
            name += "_same_test_set"

        name += f"_random_state_{self.random_state}"
        name += f"_{self.sample_type}"
        name += "_dataset.pkl"

        return name

    def log_normalize(self, x, x_max=None, x_min=None, use_torch=False):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        
        # print(x)
        # print(self.x_min_cuda)
        # print(self.x_max_cuda)
        
        if use_torch:
            return (torch.log10(x) - torch.log10(self.x_min_cuda.cuda())) / (torch.log10(self.x_max_cuda.cuda()) - torch.log10(self.x_min_cuda.cuda()))
        return (np.log10(x) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
    
    def normalize(self, x, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return (x - x_min) / (x_max - x_min)
    
    def unnormalize(self, x_normal, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return x_min + x_normal * (x_max - x_min)

    def log_unnormalize(self, x_normal, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return x_min * (x_max/x_min)**(x_normal)
    
    def balance_data(self, x, x_y_objects, x_y_shape, x_y_material, y):
        # 1st: divide into buckets
        # 2nd: sampel from buckets until balanced
        # 3rd: create augmentation instructions
        bucket_discretizer = KBinsDiscretizer(
            n_bins=self.balance_bucket,
            encode="ordinal",
            strategy="uniform",
            random_state=self.random_state
        )
        bucket_discretizer.fit(y.reshape(-1, 1))
        bucket_ids = bucket_discretizer.transform(y.reshape(-1, 1)).reshape(-1).astype(np.int64)
        #print(bucket_ids)
        bin_frequency = np.bincount(bucket_ids)
        data_indices = np.arange(len(y), dtype=np.int64)

        all_new_samples_indices = []
        #print(bin_frequency)
        # quit()

        thresh = int(self.balance_threshold * max(bin_frequency))
        #print(self.balance_threshold)
        #print(thresh)
        #quit()
        tmp_stats = []
        for i in range(len(bin_frequency)):
            if bin_frequency[i] < 1:
                continue
            
            if bin_frequency[i] > thresh:
                tmp_stats.append(bin_frequency[i])
                if self.balance_undersample:
                    # TODO: Implement undersampling
                    pass
                continue
            else:
                selection_mask = bucket_ids == i

                bin_samples_indices = data_indices[selection_mask]
                bin_objects = x_y_objects[selection_mask]

                # Setting up for balanced object sampling
                bin_objects_counter = Counter(bin_objects.tolist())
                bin_unique_objects = list(bin_objects_counter.keys())
                bin_unique_objects_count = []
                for o in bin_unique_objects:
                    bin_unique_objects_count.append(bin_objects_counter[o])
                bin_unique_objects_count = np.array(bin_unique_objects_count)

                amount_of_new_samples = thresh - bin_frequency[i]
                
                #print(f"Amount of New Samples: \n{amount_of_new_samples}")
                #print(bin_frequency[i])
                tmp_stats.append(amount_of_new_samples + bin_frequency[i])
                print(bin_unique_objects_count)
                if len(bin_unique_objects_count) == 1:
                    new_samples_per_object = np.array([amount_of_new_samples], dtype=np.int64)
                else:
                    new_samples_per_object = np.zeros(len(bin_unique_objects_count), dtype=np.int64)
                    for l in range(amount_of_new_samples):
                        new_samples_per_object[np.argmin(new_samples_per_object + bin_unique_objects_count)] += 1

                    #new_samples_per_object = 1 - (bin_unique_objects_count / bin_frequency[i]) # Get weights
                    #new_samples_per_object = new_samples_per_object / new_samples_per_object.sum() # Get density
                    #new_samples_per_object = new_samples_per_object * amount_of_new_samples
                    #new_samples_per_object = np.round(new_samples_per_object).astype(np.int64)
                print(new_samples_per_object)
                # Go through objects and sample required amount of data
                #print(bin_unique_objects)
                for k in range(len(bin_unique_objects)):
                    o = bin_unique_objects[k]
                    #print(o)
                    bin_object_mask = bin_objects == o
                    sampable_data = bin_samples_indices[bin_object_mask]
                    #print(sampable_data)

                    sample_indices = self.numpy_generator.choice(sampable_data, new_samples_per_object[k], replace=True)
                    #print(sample_indices)
                    all_new_samples_indices.append(sample_indices)
                #print(f"New Samples for Bin: \n{all_new_samples_indices}")
        all_new_samples_indices = np.concatenate(all_new_samples_indices, dtype=np.int64)
        print(np.array(tmp_stats))
        #quit()
        # Generate data augmentation instructions
        old_data_instructions = [[] for j in range(len(x))]
        new_data_instructions = []
        augmentations = ["flip_x", "flip_y", "gaussian_noise", "color_jitter"]
        if self.use_width:
            augmentations = augmentations +  ["offset"]

        
        for i in range(len(all_new_samples_indices)):
            amount_of_augmentations = self.numpy_generator.integers(1, len(augmentations) + 1)
            d_augmentations = self.numpy_generator.choice(augmentations,amount_of_augmentations, replace=False).tolist()
            new_data_instructions.append(d_augmentations)

        augmentation_instructions = new_data_instructions
        x = x[all_new_samples_indices]
        x_y_objects = x_y_objects[all_new_samples_indices]
        x_y_shape = x_y_shape[all_new_samples_indices]
        x_y_material = x_y_material[all_new_samples_indices]
        y = y[all_new_samples_indices]

        tmp_bucket_ids = bucket_discretizer.transform(y.reshape(-1, 1)).reshape(-1).astype(np.int64)
        tmp_bin_frequency = np.bincount(tmp_bucket_ids, minlength=len(bin_frequency))
        #print(tmp_bin_frequency)
        print(tmp_bin_frequency + bin_frequency)
        #quit()
        return x, x_y_objects, x_y_shape, x_y_material, y, augmentation_instructions, all_new_samples_indices
    
    def width_transform(self, x_widths):
        noise_amplitude = min(
            1 - x_widths.max(),
            x_widths.min()
        )
        
        if self.numpy_generator.random() > 0.5:
            x_widths += noise_amplitude * self.numpy_generator.random()
        else:
            x_widths -= noise_amplitude * self.numpy_generator.random()
        
        return x_widths
                    
    def prepare_data(self) -> None:
        print("Dataset File:")
        print(self.full_dataset_file)
        if self.full_dataset_file.exists() and not self.overwrite_file:
            print("Dataset already exists")
            return
        else:
            print("Creating dataset")
            # Read CSV files with objects and labels tabulated
            self.make_object_attributes()
            
            ############ Data Filtering ############

            # Extract object names as keys from data
            object_names = self.object_to_modulus.keys()
            if self.rubber_only:
                object_names = [x for x in object_names if self.object_to_material[x] == 'Rubber']

            if self.remove_paper:
                object_names = [x for x in object_names if self.object_to_material[x] != 'Paper']
            if self.exclude is  not None:
                object_names = [x for x in object_names if (x not in self.exclude)]

            if self.exclude_shape is not None:
                object_names = [x for x in object_names if self.object_to_shape[x] not in self.exclude_shape]

            if self.compliance is not None:
                assert self.compliance in ['low', 'high', 'lower_sensor', 'higher_sensor','higher_sensor_restricted', 'high_restricted', 'low_restricted']
                if self.compliance == 'low':
                    object_names = [x for x in object_names if self.object_to_material[x] in ['Rubber', 'Foam', 'Food']]
                elif self.compliance == 'high':
                    object_names = [x for x in object_names if self.object_to_material[x] not in ['Rubber', 'Foam', 'Food']]
                elif self.compliance == 'lower_sensor':
                    object_names = [x for x in object_names if self.object_to_modulus[x] < 0.275e6]
                elif self.compliance == 'higher_sensor':
                    object_names = [x for x in object_names if self.object_to_modulus[x] >= 0.275e6]
                elif self.compliance == 'higher_sensor_restricted':
                    object_names = [x for x in object_names if 0.275e8 >= self.object_to_modulus[x] >= 0.275e6]
                elif self.compliance == 'high_restricted':
                    object_names = [x for x in object_names if self.object_to_modulus[x] >= 1e9]
                elif self.compliance == 'low_restricted':
                    object_names = [x for x in object_names if self.object_to_modulus[x] <= 1e6]
                elif self.compliance == 'rolling_window':
                    object_names = [x for x in object_names if (1e3 * 10**self.rolling_window) < self.object_to_modulus[x] <= (1e6 * 10**self.rolling_window)]


            # Extract corresponding elastic modulus labels for each object
            elastic_moduli = [self.object_to_modulus[x] for x in object_names]
            print(f"Modulus Min: {np.min(elastic_moduli)}\nModulus Min: {np.max(elastic_moduli)}\nSensor Modulus: {0.275e6}")
            # quit()
            del elastic_moduli

            # Get all the paths to grasp data within directory
            paths_to_files = []
            list_files(f'{self.data_dir}/{self.training_data_folder}', paths_to_files, self.config)
            self.paths_to_files = paths_to_files
            print(len(self.paths_to_files))

            # Remove those with no estimation
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                if os.path.isfile(os.path.dirname(file_path) + '/hertz_estimate.pkl'):
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

            # Remove those with no force change
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                with open(os.path.dirname(file_path) + '/forces.pkl', 'rb') as file:
                    F = pickle.load(file)
                if F[-1] > F[0]:
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

            # Remove those with no width change
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                with open(os.path.dirname(file_path) + '/widths.pkl', 'rb') as file:
                    w = pickle.load(file)
                if w[-1] < w[0]:
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

            print(len(self.paths_to_files))
            # quit()
            # Remove those where depth is not monotonically increasing
            clean_paths_to_files = []
            marker_str = '_markers' if self.use_markers=="only_markers" else ''
            for file_path in self.paths_to_files:
                with open(os.path.dirname(file_path) + f'/depth{marker_str}.pkl', 'rb') as file:
                    depth = pickle.load(file)
                if depth[-1].max() > depth[-2].max():
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files
            print(len(self.paths_to_files))
            ##############################################################################################################
            ##############################################################################################################

            # Create data loaders based on training / validation break-up
            # Divide paths up into training and validation data
            
            objects_train = []
            objects_test = []
            x,y = [], []


            x_y_objects = []
            x_y_shape = []
            x_y_material = []
            for file_path in self.paths_to_files:
                folders = file_path.split('/')
                object_name = folders[folders.index(self.training_data_folder) + 1]
                if object_name in self.exclude: continue

                if object_name in object_names:
                    x.append(file_path)
                    y.append(self.normalization_function(self.object_to_modulus[object_name]))
                    x_y_objects.append(object_name)
                    x_y_shape.append(self.object_to_shape[object_name])
                    x_y_material.append(self.object_to_material[object_name])
                    # counter += 1

            

            # Makes sure that the test set is always the same and unseen or seen is correctly implemented
            splits = {}

            x = np.array(x)
            y = np.array(y)
            x_y_objects = np.array(x_y_objects)
            x_y_shape = np.array(x_y_shape)
            x_y_material = np.array(x_y_material)
            object_names = np.array(object_names)

            ######### Data Balancing (Only Before) #########
            if self.full_dataset_undersample:
                if self.full_dataset_undersample_amount < len(y):
                    print("Undersampling Full Dataset")
                    indices = np.arange(len(y))
                    indices = self.numpy_generator.choice(indices, self.full_dataset_undersample_amount, replace=False)
                    x = x[indices]
                    x_y_objects = x_y_objects[indices]
                    x_y_shape = x_y_shape[indices]
                    x_y_material = x_y_material[indices]
                    y = y[indices]

            if self.should_balance and self.balance_position == "before":
                print("Balacing Before")
                new_x, new_x_y_objects, new_x_y_shape, new_x_y_material, new_y, new_augmentation_instructions, all_new_samples_old_indices = self.balance_data(x, x_y_objects, x_y_shape, x_y_material, y)
                augmentation_instructions = [[] for j in range(len(y))] + new_augmentation_instructions
                x = np.concatenate([x, new_x])
                x_y_objects = np.concatenate([x_y_objects, new_x_y_objects])
                x_y_shape = np.concatenate([x_y_shape, new_x_y_shape])
                x_y_material = np.concatenate([x_y_material, new_x_y_material])
                y = np.concatenate([y, new_y])

            ############## Data Splitting ##############

            if self.sample_type == "shape_material":
                print("Splitting Strategy: Stratified")
                # First split test set. ALWAYS with random state 0 -> same test set for all
                ssp = StratifiedShuffleSplit(1, test_size=0.2, random_state=self.random_state)
                if self.use_cross_validation:
                    ksp = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
                else:
                    ksp = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=self.random_state)
                
                if self.val_on_seen_objects:
                    print("Test Data: Seen")
                    # Generate Class-Labes (shape, material) for stratified split
                    if self.stratify_with_magnitude:
                        class_labels = np.array([f"{s}_{m}_{int(np.floor(np.log10(self.object_to_modulus[o])))}" for (s, m, o) in zip(x_y_shape, x_y_material, x_y_objects)])
                    else:
                        class_labels = np.array([f"{s}_{m}" for (s, m) in zip(x_y_shape, x_y_material)])
                    print(class_labels)
                    if self.same_test_set:
                        train_indices, test_indices = next(ssp.split(y, class_labels))
                        for i, (train_split, val_split) in enumerate(ksp.split(train_indices, class_labels[train_indices])):
                            splits[i] = {
                                "train": train_indices[train_split],
                                "val": train_indices[val_split],
                                "test": test_indices # Always the same test set
                            }
                    else:
                        for i in range(self.n_splits):
                            ssp = StratifiedShuffleSplit(1, test_size=0.2, random_state=self.random_state + i) # Define different random state for each split
                            train_indices, test_indices = next(ssp.split(y, class_labels))
                            train_split, val_split = next(ssp.split(train_indices, class_labels[train_indices]))
                            splits[i] = {
                                "train": train_indices[train_split],
                                "val": train_indices[val_split],
                                "test": test_indices
                            }
                else:
                    print("Test Data: Unseen")
                    # Get Test set indices from x and y
                    if self.stratify_with_magnitude:
                        class_labels = np.array([f"{self.object_to_shape[o]}_{self.object_to_material[o]}_{int(np.floor(np.log10(self.object_to_modulus[o])))}" for o in object_names])
                    else:
                        class_labels = np.array([f"{self.object_to_shape[o]}_{self.object_to_material[o]}" for o in object_names])
                    # Determine objects for test set and train/validation set
                    label_counter = Counter(class_labels)
                    splitable_objects_mask = np.array([label_counter[o] for o in class_labels]) > 2
                    unsplitable_objects_mask = np.array([label_counter[o] for o in class_labels]) <= 2 # TODO leave out?
                    if self.same_test_set:
                        train_object_indices, test_object_indices = next(ssp.split(object_names[splitable_objects_mask], class_labels[splitable_objects_mask]))

                        objects_test = np.array(object_names[splitable_objects_mask][test_object_indices])
                        objects_train = np.array(object_names[splitable_objects_mask][train_object_indices])
                        
                        # Extract all indices of datapoints contained inside test objects
                        train_indices, test_indices = [], []
                        for i in range(len(x)):
                            if x_y_objects[i] in objects_test:
                                test_indices.append(i)
                            elif x_y_objects[i] in objects_train: # making sure that unstratifiable objects are not used
                                train_indices.append(i)
                        
                        train_indices = np.array(train_indices)
                        test_indices = np.array(test_indices)
                        # Split train objects into train and validation folds
                        for fold, ((train_split, val_split)) in enumerate(ksp.split(objects_train, class_labels[splitable_objects_mask][train_object_indices])):
                            # Extract indices for specific object that should be train or val
                            #print(f"Fold {fold}")
                            #print(train_split)
                            #print(val_split)

                            t_i, v_i = [], []
                            # Get indices of datapoints contained inside train and validation folds
                            for i in train_indices:
                                if x_y_objects[i] in objects_train[train_split]:
                                    t_i.append(i)
                                else:
                                    v_i.append(i)
                            
                            splits[fold] = {
                                "train": t_i,
                                "val": v_i,
                                "test": test_indices # Always the same test set
                            }
                    else:
                        for j in range(self.n_splits):
                            ssp = StratifiedShuffleSplit(1, test_size=0.2, random_state=self.random_state + i) # Define different random state for each split
                            train_object_indices, test_object_indices = next(ssp.split(object_names[splitable_objects_mask], class_labels[splitable_objects_mask]))

                            objects_test = np.array(object_names[splitable_objects_mask][test_object_indices])
                            objects_train = np.array(object_names[splitable_objects_mask][train_object_indices])
                            
                            # Extract all indices of datapoints contained inside test objects
                            train_indices, test_indices = [], []
                            for i in range(len(x)):
                                if x_y_objects[i] in objects_test:
                                    test_indices.append(i)
                                elif x_y_objects[i] in objects_train: # making sure that unstratifiable objects are not used
                                    train_indices.append(i)
                            
                            train_indices = np.array(train_indices)
                            test_indices = np.array(test_indices)
                            # Split train objects into train and validation folds
                            train_split, val_split = next(ssp.split(objects_train, class_labels[splitable_objects_mask][train_object_indices]))
                            t_i, v_i = [], []
                            # Get indices of datapoints contained inside train and validation folds
                            for i in train_indices:
                                if x_y_objects[i] in objects_train[train_split]:
                                    t_i.append(i)
                                else:
                                    v_i.append(i)
                                
                            splits[j] = {
                                "train": t_i,
                                "val": v_i,
                                "test": test_indices
                            }
            else:
                print("Splitting Strategy: Random")
                # First split test set. ALWAYS with random state 0 -> same test set for all
                ssp = ShuffleSplit(1, test_size=0.2, random_state=self.random_state)
                if self.use_cross_validation:
                    ksp = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
                else:
                    ksp = ShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=self.random_state)

                if self.val_on_seen_objects:
                    print("Test Data: Seen")
                    if self.same_test_set:
                        train_indices, test_indices = next(ssp.split(y))
                        
                        for i, (train_split, val_split) in enumerate(ksp.split(train_indices)):
                            splits[i] = {
                                "train": train_indices[train_split],
                                "val": train_indices[val_split],
                                "test": test_indices # Always the same test set
                            }
                    else:
                        # TODO: implement same test set
                        for i in range(self.n_splits):
                            ssp = ShuffleSplit(1, test_size=0.2, random_state=self.random_state + i) # Define different random state for each split
                            train_indices, test_indices = next(ssp.split(y))
                            train_split, val_split = next(ssp.split(train_indices))
                            splits[i] = {
                                "train": train_indices[train_split],
                                "val": train_indices[val_split],
                                "test": test_indices
                            }
                else:
                    print("Test Data: Unseen")
                    if self.same_test_set:
                        # Determine objects for test set and train/validation set
                        train_object_indices, test_object_indices = next(ssp.split(object_names))

                        objects_test = np.array(object_names[test_object_indices])
                        objects_train = np.array(object_names[train_object_indices])
                        
                        # Extract all indices of datapoints contained inside test objects
                        train_indices, test_indices = [], []
                        for i in range(len(x)):
                            if x_y_objects[i] in objects_test:
                                test_indices.append(i)
                            else:
                                train_indices.append(i)
                        
                        train_indices = np.array(train_indices)
                        test_indices = np.array(test_indices)

                        # Split train objects into train and validation folds
                        for fold, (train_split, val_split) in enumerate(ksp.split(objects_train)):
                            # Extract indices for specific object that should be train or val
                            print(f"Fold {fold}")
                            print(train_split)
                            print(val_split)

                            t_i, v_i = [], []
                            # Get indices of datapoints contained inside train and validation folds
                            for i in train_indices:
                                if x_y_objects[i] in objects_train[train_split]:
                                    t_i.append(i)
                                else:
                                    v_i.append(i)
                            
                            splits[fold] = {
                                "train": t_i,
                                "val": v_i,
                                "test": test_indices # Always the same test set
                            }
                    else:
                        for j in range(self.n_splits):
                            ssp = ShuffleSplit(1, test_size=0.2, random_state=self.random_state + j) # Define different random state for each split
                            # Determine objects for test set and train/validation set
                            train_object_indices, test_object_indices = next(ssp.split(object_names))

                            objects_test = np.array(object_names[test_object_indices])
                            objects_train = np.array(object_names[train_object_indices])
                            
                            # Extract all indices of datapoints contained inside test objects
                            train_indices, test_indices = [], []
                            for i in range(len(x)):
                                if x_y_objects[i] in objects_test:
                                    test_indices.append(i)
                                else:
                                    train_indices.append(i)
                            
                            train_indices = np.array(train_indices)
                            test_indices = np.array(test_indices)

                            train_split, val_split = next(ssp.split(objects_train))
                            
                            # Extract indices for specific object that should be train or val
                            print(f"Fold {j}")
                            print(train_split)
                            print(val_split)

                            t_i, v_i = [], []
                            # Get indices of datapoints contained inside train and validation folds
                            for i in train_indices:
                                if x_y_objects[i] in objects_train[train_split]:
                                    t_i.append(i)
                                else:
                                    v_i.append(i)
                                
                            splits[j] = {
                                "train": t_i,
                                "val": v_i,
                                "test": test_indices
                            }
            
            ########## Balancing Data (Only After) ##########
            if self.should_balance:
                if self.balance_position == "after_all_folds":
                    print("Balacing after splitting but combining each split")
                    start_i = len(y)
                    new_x, new_x_y_objects, new_x_y_shape, new_x_y_material, new_y, new_augmentation_instructions, all_new_samples_old_indices = self.balance_data(x, x_y_objects, x_y_shape, x_y_material, y)

                    augmentation_instructions = [[] for j in range(len(y))] + new_augmentation_instructions
                    x = np.concatenate([x, new_x])
                    x_y_objects = np.concatenate([x_y_objects, new_x_y_objects])
                    x_y_shape = np.concatenate([x_y_shape, new_x_y_shape])
                    x_y_material = np.concatenate([x_y_material, new_x_y_material])
                    y = np.concatenate([y, new_y])

                    new_indices = np.arange(start_i, len(y))

                    for i in range(self.n_splits):
                        # only include the indices of the current fold, skipping other generated augmentations, but increased memory usage                       
                        for sp in ["train", "val", "test"]:
                            indices = splits[i][sp]
                            relevant_indices_mask = np.isin(all_new_samples_old_indices, indices)
                            splits[i][sp] = np.concatenate([indices, new_indices[relevant_indices_mask]])

                elif self.balance_position == "after_isolated":
                    print("Balance Data for each split isolated")
                    augmentation_instructions = [[] for j in range(len(y))]
                    for i in range(self.n_splits):
                        # only include the indices of the current fold, skipping other generated augmentations, but increased memory usage                       
                        for sp in ["train", "val", "test"]:
                            indices = splits[i][sp]
                            start_i = len(y)
                            new_x, new_x_y_objects, new_x_y_shape, new_x_y_material, new_y, new_augmentation_instructions, all_new_samples_old_indices = self.balance_data(x[indices], x_y_objects[indices], x_y_shape[indices], x_y_material[indices], y[indices])
                            augmentation_instructions = augmentation_instructions + new_augmentation_instructions
                            x = np.concatenate([x, new_x])
                            x_y_objects = np.concatenate([x_y_objects, new_x_y_objects])
                            x_y_shape = np.concatenate([x_y_shape, new_x_y_shape])
                            x_y_material = np.concatenate([x_y_material, new_x_y_material])
                            y = np.concatenate([y, new_y])
                            splits[i][sp] = np.concatenate([indices, np.arange(start_i, len(y))])
                elif self.balance_position == "after_isolated_lower_memory":
                    print("Balance Data for each split isolated")
                    augmentation_instructions = [] # No prefabricated augmentations
                    for fold in range(self.n_splits):
                        # only include the indices of the current fold, skipping other generated augmentations, but increased memory usage                       
                        all_splits = ["train"]
                        if self.balance_validation_set:
                            all_splits.append("val")
                        if self.balance_test_set:
                            all_splits.append("test")
                        for sp in all_splits:
                            indices = np.array(splits[fold][sp])
                            new_x, new_x_y_objects, new_x_y_shape, new_x_y_material, new_y, new_augmentation_instructions, all_new_samples_old_indices = self.balance_data(x[indices], x_y_objects[indices], x_y_shape[indices], x_y_material[indices], y[indices])
                            #augs[i][sp] = [[] for _ in range(len(indices))] + new_augmentation_instructions
                            print(f"New Samples for {fold}:")
                            print(type(indices))
                            print(type(all_new_samples_old_indices))
                            print(indices[all_new_samples_old_indices.astype(int)])
                            splits[fold][sp] = np.concatenate([indices, indices[all_new_samples_old_indices]]) # only store old data points. Augmentations are generated on the fly!
                            

            print("Loading data into memory")
            all_data = self.get_all_data(DataLoader(CustomDataset(self.config, x.tolist(), y.tolist(), self.normalization_values, validation_dataset=True), shuffle=False))
            
            ########## Apply Data Augmentations (Not for _lower_memory version (so that the data does not need to be stored))##########
            if self.should_balance:
                x_flip = torchvision.transforms.RandomHorizontalFlip(1.0)      
                y_flip = torchvision.transforms.RandomVerticalFlip(1.0)
                print("Running Augmentations")
                for i in tqdm.tqdm(range(len(augmentation_instructions))):
                    for instruction in augmentation_instructions[i]:
                        if instruction == "offset":
                            all_data["x_widths"][i] = self.width_transform(all_data["x_widths"][i])
                        elif instruction == "flip_x":
                           all_data["x_frames"][i] = x_flip(all_data["x_frames"][i])
                        elif instruction == "flip_y":
                            all_data["x_frames"][i] = y_flip(all_data["x_frames"][i])


            dataset = {
                "all_data": all_data,
                "splits": splits,
            }
            # quit()

            print("Storing data")
            pickle.dump(dataset, self.full_dataset_file.open('wb'))
            del dataset
            del all_data
            print("Finished")

            return

    def make_object_attributes(self):
        self.object_names = []
        self.object_to_modulus = {}
        self.object_to_material = {}
        self.object_to_shape = {}
            
        self.object_material = []
        self.object_shape = []
            
        csv_file_path = f'{self.data_dir}/{self.labels_csv_name}'
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) # Skip title row
            for row in csv_reader:
                if row[self.csv_modulus_column] != '' and float(row[self.csv_modulus_column]) > 0:
                    modulus = float(row[self.csv_modulus_column])
                        
                    self.object_to_modulus[row[1]] = modulus
                    self.object_to_material[row[1]] = row[self.csv_material_column]
                    self.object_to_shape[row[1]] = row[self.csv_shape_column]
                        
                    if row[1] not in self.exclude:
                        self.object_names.append(row[1])
                        self.object_material.append(row[3])
                        self.object_shape.append(row[2])

    def get_all_data(self, data):
        all_data = {
            "x_frames": [],
            "x_forces": [],
            "x_widths": [],
            "x_estimations": [],
            "y_label": [],
            "object_name": []
        }
        for batch in tqdm.tqdm(data):
            x_frames, x_forces, x_widths, x_estimations, y_label, object_name = batch
            all_data["x_frames"].append(x_frames.cpu()[0])
            all_data["x_forces"].append(x_forces.cpu()[0])
            all_data["x_widths"].append(x_widths.cpu()[0])
            all_data["x_estimations"].append(x_estimations.cpu()[0])
            all_data["y_label"].append(y_label.cpu()[0])
            all_data["object_name"].append(object_name[0])

        all_data["x_frames"] = torch.stack(all_data["x_frames"])
        all_data["x_forces"] = torch.stack(all_data["x_forces"])
        all_data["x_widths"] = torch.stack(all_data["x_widths"])
        all_data["x_estimations"] = torch.stack(all_data["x_estimations"])
        all_data["y_label"] = torch.stack(all_data["y_label"])
        all_data["object_name"] = all_data["object_name"]
        return all_data
    
    def setup(self, stage: str=None, fold=0) -> None:
        
        assert fold < self.n_splits # Making sure that the fold actually exists 
        print("Loading data")
        self.dataset = pickle.load(self.full_dataset_file.open('rb'))
        # TODO: test set with data augmentation?
        if self.config["use_width_transforms"]:
            self.train_indices = torch.cat([torch.tensor(self.dataset["splits"][fold]["train"]), torch.tensor(np.array(self.dataset["splits"][fold]["train"]) + len(self.dataset["all_data"]["x_frames"])//2)])
            self.val_indices = torch.cat([torch.tensor(self.dataset["splits"][fold]["val"]), torch.tensor(np.array(self.dataset["splits"][fold]["val"]) + len(self.dataset["all_data"]["x_frames"])//2)])
            self.test_indices = torch.cat([torch.tensor(self.dataset["splits"][fold]["test"]), torch.tensor(np.array(self.dataset["splits"][fold]["test"]) + len(self.dataset["all_data"]["x_frames"])//2)])
            
            #print(self.train_indices.tolist())
            #print(self.val_indices.tolist())
            #print(self.test_indices.tolist())

            #assert torch.all(train_indices != val_indices)
            #assert torch.all(val_indices != test_indices)
            #assert torch.all(train_indices != test_indices)

            print(f"Train: {len(self.train_indices)}\nVal: {len(self.val_indices)}\nTest: {len(self.test_indices)}")

            self.dataset = {
                "train": {
                    "x_frames": self.dataset["all_data"]["x_frames"][self.train_indices],
                    "x_forces": self.dataset["all_data"]["x_forces"][self.train_indices],
                    "x_widths": self.dataset["all_data"]["x_widths"][self.train_indices],
                    "x_estimations": self.dataset["all_data"]["x_estimations"][self.train_indices],
                    "y_label": self.dataset["all_data"]["y_label"][self.train_indices],
                    "object_name": [self.dataset["all_data"]["object_name"][o] for o in self.dataset["splits"][fold]["train"]] + [self.dataset["all_data"]["object_name"][len(self.dataset["all_data"]["object_name"])//2:][o] for o in self.dataset["splits"][fold]["train"]]
                },
                "val": {
                    "x_frames": self.dataset["all_data"]["x_frames"][self.val_indices],
                    "x_forces": self.dataset["all_data"]["x_forces"][self.val_indices],
                    "x_widths": self.dataset["all_data"]["x_widths"][self.val_indices],
                    "x_estimations": self.dataset["all_data"]["x_estimations"][self.val_indices],
                    "y_label": self.dataset["all_data"]["y_label"][self.val_indices],
                    "object_name": [self.dataset["all_data"]["object_name"][o] for o in self.dataset["splits"][fold]["val"]] + [self.dataset["all_data"]["object_name"][len(self.dataset["all_data"]["object_name"])//2:][o] for o in self.dataset["splits"][fold]["val"]]
                },
                "test": {
                    "x_frames": self.dataset["all_data"]["x_frames"][self.test_indices],
                    "x_forces": self.dataset["all_data"]["x_forces"][self.test_indices],
                    "x_widths": self.dataset["all_data"]["x_widths"][self.test_indices],
                    "x_estimations": self.dataset["all_data"]["x_estimations"][self.test_indices],
                    "y_label": self.dataset["all_data"]["y_label"][self.test_indices],
                    "object_name": [self.dataset["all_data"]["object_name"][o] for o in self.dataset["splits"][fold]["test"]] + [self.dataset["all_data"]["object_name"][len(self.dataset["all_data"]["object_name"])//2:][o] for o in self.dataset["splits"][fold]["test"]]
                },
            }
        else:
            self.train_indices = self.dataset["splits"][fold]["train"]
            self.val_indices = self.dataset["splits"][fold]["val"]
            self.test_indices = self.dataset["splits"][fold]["test"]
            #print(train_indices)
            #print(val_indices)
            #print(test_indices)

            print(f"Train: {len(self.train_indices)}\nVal: {len(self.val_indices)}\nTest: {len(self.test_indices)}")

            #assert torch.all(torch.tensor(self.train_indices) != torch.tensor(self.val_indices))
            #assert torch.all(torch.tensor(self.val_indices) != torch.tensor(self.test_indices))
            #assert torch.all(torch.tensor(self.train_indices) != torch.tensor(self.test_indices))


            self.dataset = {
                "train": {
                    "x_frames": self.dataset["all_data"]["x_frames"][self.train_indices],
                    "x_forces": self.dataset["all_data"]["x_forces"][self.train_indices],
                    "x_widths": self.dataset["all_data"]["x_widths"][self.train_indices],
                    "x_estimations": self.dataset["all_data"]["x_estimations"][self.train_indices],
                    "y_label": self.dataset["all_data"]["y_label"][self.train_indices],
                    "object_name": [self.dataset["all_data"]["object_name"][o] for o in self.train_indices]
                },
                "val": {
                    "x_frames": self.dataset["all_data"]["x_frames"][self.val_indices],
                    "x_forces": self.dataset["all_data"]["x_forces"][self.val_indices],
                    "x_widths": self.dataset["all_data"]["x_widths"][self.val_indices],
                    "x_estimations": self.dataset["all_data"]["x_estimations"][self.val_indices],
                    "y_label": self.dataset["all_data"]["y_label"][self.val_indices],
                    "object_name": [self.dataset["all_data"]["object_name"][o] for o in self.val_indices]
                },
                "test": {
                    "x_frames": self.dataset["all_data"]["x_frames"][self.test_indices],
                    "x_forces": self.dataset["all_data"]["x_forces"][self.test_indices],
                    "x_widths": self.dataset["all_data"]["x_widths"][self.test_indices],
                    "x_estimations": self.dataset["all_data"]["x_estimations"][self.test_indices],
                    "y_label": self.dataset["all_data"]["y_label"][self.test_indices],
                    "object_name": [self.dataset["all_data"]["object_name"][o] for o in self.test_indices]
                },
            }

        # self.dataset = torch.load(self.full_dataset_file)

        return super().setup(stage)
    
    def train_dataloader(self, batch_size=None, apply_augmentations=True):
        if batch_size is None:
            batch_size = self.batch_size
        
        return DataLoader(YoungsDataset(self.dataset["train"], self.balance_position=="after_isolated_lower_memory" and apply_augmentations), batch_size=batch_size, shuffle=True, num_workers=self.worker)
    
    def val_dataloader(self, batch_size=None, apply_augmentations=True):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(YoungsDataset(self.dataset["val"], self.balance_position=="after_isolated_lower_memory" and apply_augmentations), batch_size=batch_size, shuffle=False, num_workers=self.worker)
    
    def test_dataloader(self, batch_size=None, apply_augmentations=True):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(YoungsDataset(self.dataset["test"], (self.balance_position=="after_isolated_lower_memory" and self.balance_test_set and apply_augmentations)), batch_size=batch_size, shuffle=False, num_workers=self.worker)
    
if __name__ == "__main__":
    data_module = KFoldYoungsDataModule(
        data_dir='/home/malte.kuhlmann/youngs-modulus/data',
        training_data_folder="gelsight_youngs_modulus_dataset",
        sample_type='random',
        use_estimations=True,
        use_force=True,
        use_width=True,
        use_width_transforms=False,
        val_on_seen_objects=True,
        exclude= [
            'playdoh',
            'silly_puty', 
            'blue_sponge_dry', 
            'blue_sponge_wet', 
            'apple',
            'orange',
            'strawberry',
            'ripe_banana',
            'unripe_banana', 
            'lacrosse_ball', 
            'baseball', 
            'racquet_ball', 
            'tennis_ball'
        ],
        overwrite_file=True,
    )
    data_module.prepare_data()
    data_module.setup()
    for batch in data_module.test_dataloader():
        print(batch)
        break

    data_module.setup(fold=5)
    for batch in data_module.test_dataloader():
        print(batch)
        break