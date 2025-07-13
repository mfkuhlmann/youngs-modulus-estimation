from collections import defaultdict
import csv
import os
from pathlib import Path
import pickle
import random
import lightning
from sklearn.model_selection import train_test_split
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm

from dataset.offloaded_dataset import CustomDataset
from dataset.utils import list_files


class YoungsDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self):
        return len(self.data["x_frames"])
    
    def __getitem__(self, idx):
        return self.data["x_frames"][idx], self.data["x_forces"][idx], self.data["x_widths"][idx], self.data["x_estimations"][idx], self.data["y_label"][idx], self.data["object_name"][idx]
    



class YoungsDataModule(lightning.LightningDataModule):
    def __init__(self, data_dir, training_data_folder, sample_type, exclude, overwrite_file=False, worker=0, batch_size=1, val_on_seen_objects=False, rubber_only=False, use_estimations=False, use_force=False, use_width=False, use_width_transforms=False, labels_csv_name='dataset_objects_and_compliance.csv', csv_modulus_column=14, csv_shape_column=2, csv_material_column=3, image_style="RGB", use_markers=False, random_state = 42) -> None:
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
        self.rubber_only = rubber_only
        self.val_on_seen_objects = val_on_seen_objects
        self.use_force = use_force
        self.use_width = use_width
        self.use_estimations = use_estimations
        self.labels_csv_name = labels_csv_name
        self.csv_modulus_column = csv_modulus_column
        self.csv_shape_column = csv_shape_column
        self.csv_material_column = csv_material_column
        self.exclude = exclude
        self.image_style = image_style
        self.use_markers = use_markers
        self.random_state = random_state
        self.config = {
            "img_style": image_style,
            "use_markers": use_markers,
            "training_data_folder": training_data_folder,
            "use_force": use_force,
            "use_width": use_width,
            "use_estimations": use_estimations,
            "use_width_transforms": use_width_transforms
        }

                # Create max values for scaling
        self.normalization_values = { # Based on acquired data maximums
            'min_modulus': 1e3,
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

        print(self.full_dataset_file)
        #print(self.full_dataset_file.exists())
        #print(self.val_on_seen_objects)
        #quit()
        random.seed(self.random_state)

    def stratified_sample_and_remainder(self, data_list, category1, category2, ratio):
        combined_indices = defaultdict(list)
        for index, (cat1, cat2) in enumerate(zip(category1, category2)):
            combined_indices[(cat1, cat2)].append(index)

        sampled_data = []
        sampled_category1 = []
        sampled_category2 = []
        remainder_data = []
        remainder_category1 = []
        remainder_category2 = []

        for (cat1, cat2), indices in combined_indices.items():
            sample_size = int(len(indices) * ratio)
            sampled_indices = random.sample(indices, sample_size) # TODO: Add random state

            for idx in sampled_indices:
                sampled_data.append(data_list[idx])
                sampled_category1.append(category1[idx])
                sampled_category2.append(category2[idx])

            remainder_indices = set(indices) - set(sampled_indices)
            for idx in remainder_indices:
                remainder_data.append(data_list[idx])
                remainder_category1.append(category1[idx])
                remainder_category2.append(category2[idx])
    
        return sampled_data, sampled_category1, sampled_category2, remainder_data, remainder_category1, remainder_category2

    def get_dataset_name(self):
        name = "dataset"

        if self.rubber_only:
            name += "_rubber_only"

        name += f"_{self.sample_type}"
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

        if self.use_markers:
            name += "_only_markers"

        name += f"_random_state_{self.random_state}"
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

    def log_unnormalize(self, x_normal, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return x_min * (x_max/x_min)**(x_normal)
    
    def prepare_data(self) -> None:
        
        if self.full_dataset_file.exists() and not self.overwrite_file:
            print("Dataset already exists")
            return
        else:
            print("Creating dataset")
            # Data structures for training
            self.object_names = []
            self.object_to_modulus = {}

            # Read CSV files with objects and labels tabulated
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
                        

            
            # Extract object names as keys from data
            object_names = self.object_to_modulus.keys()
            if self.rubber_only:
                object_names = [x for x in object_names if self.object_to_material[x] == 'Rubber']
            else:
                object_names = [x for x in object_names if (x not in self.exclude)]

            # Extract corresponding elastic modulus labels for each object
            elastic_moduli = [self.object_to_modulus[x] for x in object_names]

            
            # Split objects into validation or training
            if self.sample_type == "shape_material":
                print("shape_material")
                sampled_objects_train, train_category1, train_category2, remainder_objects_val_all, val_all_category1, val_all_category2 = self.stratified_sample_and_remainder(self.object_names, self.object_shape, self.object_material, ratio = 0.6)
                self.objects_train = sampled_objects_train
                self.objects_val = remainder_objects_val_all
                
                val_objects, val_category1, val_category2, test_objects, test_category1, test_category2 = self.stratified_sample_and_remainder(remainder_objects_val_all, val_all_category1, val_all_category2, ratio = 0.5)
                self.objects_val = val_objects
                self.objects_test = test_objects
                
            else:
                print("random")
                # self.objects_train, self.objects_val, _, _ = train_test_split(object_names, elastic_moduli, test_size=self.val_pct, random_state=self.random_state)
                self.objects_train, self.objects_val_all, _, _ = train_test_split(object_names, elastic_moduli, test_size=0.4, random_state=self.random_state)
                print(len(self.objects_train), len(self.objects_val_all))
                
                elastic_moduli_test = [self.object_to_modulus[x] for x in self.objects_val_all]
                
                self.objects_val, self.objects_test, _, _ = train_test_split(self.objects_val_all, elastic_moduli_test, test_size=0.5, random_state=self.random_state)
                
                print(len(self.objects_train), len(self.objects_val),len(self.objects_test))

            print(self.objects_train)
            print(self.objects_val)
            print(self.objects_test)    
            # quit()
            del object_names, elastic_moduli

            # Get all the paths to grasp data within directory
            paths_to_files = []
            list_files(f'{self.data_dir}/{self.training_data_folder}', paths_to_files, self.config)
            self.paths_to_files = paths_to_files
            print(len(self.paths_to_files))
            # Remove those with no estimation
            if self.use_estimations:
                clean_paths_to_files = []
                for file_path in self.paths_to_files:
                    if os.path.isfile(os.path.dirname(file_path) + '/hertz_estimate.pkl'):
                        clean_paths_to_files.append(file_path)
                self.paths_to_files = clean_paths_to_files

            # Remove those with no force change
            if self.use_force:
                clean_paths_to_files = []
                for file_path in self.paths_to_files:
                    with open(os.path.dirname(file_path) + '/forces.pkl', 'rb') as file:
                        F = pickle.load(file)
                    if F[-1] > F[0]:
                        clean_paths_to_files.append(file_path)
                self.paths_to_files = clean_paths_to_files

            # Remove those with no width change
            if self.use_width:
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
            marker_str = '_markers' if self.use_markers else ''
            for file_path in self.paths_to_files:
                with open(os.path.dirname(file_path) + f'/depth{marker_str}.pkl', 'rb') as file:
                    depth = pickle.load(file)
                if depth[-1].max() > depth[-2].max():
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

            ##############################################################################################################
            ##############################################################################################################

            # Create data loaders based on training / validation break-up
            # Divide paths up into training and validation data
            x_train, y_train= [], []
            x_val, y_val = [], []
            x_test, y_test = [], []
            self.object_names = []
            for file_path in self.paths_to_files:
                folders = file_path.split('/')
                object_name = folders[folders.index(self.training_data_folder) + 1]
                if object_name in self.exclude: continue

                if object_name in self.objects_train:
                    self.object_names.append(object_name)
                    x_train.append(file_path)
                    y_train.append(self.log_normalize(self.object_to_modulus[object_name]))

                elif object_name in self.objects_val:
                    self.object_names.append(object_name)
                    x_val.append(file_path)
                    y_val.append(self.log_normalize(self.object_to_modulus[object_name]))
                
                if object_name in self.objects_test:
                    self.object_names.append(object_name)
                    x_test.append(file_path)
                    y_test.append(self.log_normalize(self.object_to_modulus[object_name]))

            # For seen objects, randomly shuffle
            if self.val_on_seen_objects:
                print("Seen")
                
                x_train, x_val, y_train, y_val = train_test_split(x_train + x_val + x_test, y_train + y_val + y_test, test_size=0.4, random_state=self.random_state)
                
                x_val, x_test, y_val, y_test = train_test_split( x_val, y_val, test_size=0.5, random_state=self.random_state)

                # mix every object by shuffeling and then splitting again

            else:
                print("UnSeen")
                # x_train, x_val, y_train, y_val = train_test_split(x_train + x_val, y_train + y_val, test_size=0.25, random_state=self.random_state)
                # Test set already finished. Only shuffel val and train to mix objects.
                # If test set is unseen then we also want the val to be unseen
            
                
            ########   Set new object_train and object_val   ########
            object_val = []
            for i in x_val:
                objname = i.split('/')
                objname = objname[-4]
                if objname in object_val:
                    continue
                else:
                    object_val.append(objname)
            self.objects_val = object_val
            
            object_train = []
            for i in x_train:
                objname = i.split('/')
                objname = objname[-4]
                if objname in object_train:
                    continue
                else:
                    object_train.append(objname)
            self.objects_train = object_train
            
            object_test = []
            for i in x_test:
                objname = i.split('/')
                objname = objname[-4]
                
                if objname in object_test:
                    continue
                else:
                    object_test.append(objname)
            self.objects_test = object_test

            ########   Set new object_train and object_val   ########

            dataset = {
                "x_train": x_train,
                "x_val": x_val,
                "x_test": x_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test
            }
            
            train_data = self.get_all_data(DataLoader(CustomDataset(self.config, dataset["x_train"], dataset["y_train"], self.normalization_values, validation_dataset=False), shuffle=False))
            val_data = self.get_all_data(DataLoader(CustomDataset(self.config, dataset["x_val"], dataset["y_val"], self.normalization_values, validation_dataset=True), shuffle=False))
            test_data = self.get_all_data(DataLoader(CustomDataset(self.config, dataset["x_test"], dataset["y_test"], self.normalization_values, validation_dataset=True), shuffle=False))

            dataset = {
                "train": train_data,
                "val": val_data,
                "test": test_data
            }
            pickle.dump(dataset, self.full_dataset_file.open('wb'))
            print("Finished")
            return

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
        return all_data
    
    def setup(self, stage: str=None) -> None:

        self.dataset = pickle.load(self.full_dataset_file.open('rb'))
        # self.dataset = torch.load(self.full_dataset_file)

        return super().setup(stage)
    
    def train_dataloader(self):
        return DataLoader(YoungsDataset(self.dataset["train"]), batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
    
    def val_dataloader(self):
        return DataLoader(YoungsDataset(self.dataset["val"]), batch_size=self.batch_size, shuffle=False, num_workers=self.worker)
    
    def test_dataloader(self):
        return DataLoader(YoungsDataset(self.dataset["test"]), batch_size=self.batch_size, shuffle=False, num_workers=self.worker)
    
if __name__ == "__main__":
    data_module = YoungsDataModule(
        data_dir='/home/malte.kuhlmann/youngs-modulus/data',
        training_data_folder="gelsight_youngs_modulus_dataset",
        sample_type='random',
        use_estimations=False,
        use_force=False,
        use_width=False,
        use_width_transforms=False,
        exclude=[],#['playdoh', 'silly_puty', 'blue_sponge_dry', 'blue_sponge_wet', 
                #'apple', 'orange', 'strawberry', 'ripe_banana', 'unripe_banana', 
                #'lacrosse_ball', 'baseball', 'racquet_ball', 'tennis_ball']
    )
    data_module.prepare_data()
    data_module.setup()
    for batch in data_module.test_dataloader():
        print(batch)
        break