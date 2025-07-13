from collections import Counter
import csv
import os
from pathlib import Path
import pickle
import lightning
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import StratifiedKFold
import torchvision
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm

# from dataset.offloaded_dataset import CustomDataset
# from dataset.utils import list_files


class ShoreEstimatedYoungsDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self):
        return len(self.data["x_frames"])
    
    def __getitem__(self, idx):
        return torch.movedim(torch.tensor(self.data["x_frames"][idx], dtype=torch.float32), -1, 1)/255, torch.zeros((3, 1), dtype=torch.float32), torch.zeros((3, 1), dtype=torch.float32), torch.zeros((2,1), dtype=torch.float32), torch.tensor([self.data["y_label"][idx]], dtype=torch.float32), self.data["object_name"][idx] # making sure that the output matches with the expected pattern by the models
    
class KFoldShoreDataModule(lightning.LightningDataModule):
    def __init__(self, data_dir, folder_name, worker=1, intensity_threshold=None, seen=True, use_log_normalization=True, batch_size=32, n_splits=10, same_test_set=False, use_cross_validation=True, only_evaluation=False, random_state=0):
        super().__init__()

        self.data_dir = Path(data_dir).resolve()

        self.folder_name = folder_name
        self.intensity_threshold = intensity_threshold
        self.seen = seen
        self.batch_size = batch_size
        self.worker = worker
        self.use_log_normalization = use_log_normalization
        self.n_splits = n_splits
        self.use_cross_validation = use_cross_validation
        self.random_state = random_state
        self.same_test_set = same_test_set
        self.path_to_folds = self.data_dir / "dataset_variants" / f"{self.get_name()}_folds.pkl"
        tmp_st = "_old_norm" if only_evaluation else ""
        self.path_to_stored_data = self.data_dir / "dataset_variants" / f"shore_full_data{tmp_st}.pkl"
        self.only_evaluation = only_evaluation
        if self.only_evaluation:
            self.normalization_values = { # Based on acquired data maximums
            'min_modulus': 1e3,
            'max_modulus': 1e12,
            'min_estimate': 1e2,
            'max_estimate': 1e14,
            'max_depth': 7.0,
            'max_width': 0.08, # Measured from Franka Panda gripper
            'max_force': 60.0,
        }
        else:
            self.normalization_values = { # Based on acquired data maximums
                'min_modulus': 1e3,
                'max_modulus': 1e7,
                'min_estimate': 1e2,
                'max_estimate': 1e7,
                'max_depth': 7.0,
                'max_width': 0.08, # Measured from Franka Panda gripper
                'max_force': 60.0,
            }

        print(self.normalization_values)

        if self.use_log_normalization:
            print("Using log normalization")
            self.normalization_function = self.log_normalize
            self.unnormalization_function = self.log_unnormalize
        else:
            print("Using standard normalization")
            self.normalization_function = self.normalize
            self.unnormalization_function = self.unnormalize
    
    def get_name(self):
        name = "k_fold_shore"

        if self.seen:
            name += "_seen"
        else:
            name += "_unseen"

        if self.n_splits != 10:
            name += f"_{self.n_splits}_folds"
        
        if not self.use_cross_validation: 
            name += "_no_cross_validation"
        
        name += f"_random_state_{self.random_state}"

        name += "_data"
    
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
    

    def extract_frames_from_video(self, video_path):
        # return video
        video = cv.VideoCapture(str(video_path))
        all_frames = []
        all_intensities = []

        first_frame_idx = -1 if self.intensity_threshold is not None else 0
        idx = 0

        # TODO: Improve loading speed
        
        while True:
            success, frame = video.read()
            if not success:
                break
            all_frames.append(cv.cvtColor(cv.resize(frame, (350, 250)), cv.COLOR_BGR2RGB))
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #TODO: Check if BGR or other different format -> opencv always BGR
            mean_intensity = np.mean(gray_frame)
            all_intensities.append(mean_intensity)
            
            if self.intensity_threshold is not None:
                if first_frame_idx < -1 and mean_intensity > self.intensity_threshold:
                    first_frame_idx = idx

            idx += 1
            # highest_intensity_frame_idx = np.argmax(mean_intensity)

        video.release()
        all_intensities = np.array(all_intensities)
         # Check if shape is correct dimensions

        highest_intensity_frame_idx = np.argmax(all_intensities)

        mid_idx = (highest_intensity_frame_idx - first_frame_idx) // 2
        mid_idx = first_frame_idx + mid_idx

        return np.array([all_frames[first_frame_idx], all_frames[mid_idx], all_frames[highest_intensity_frame_idx]])

    def shore_00_to_youngs(self, shore): # From https://www.researchgate.net/publication/336239577_Can_You_Estimate_Modulus_From_Durometer_Hardness_for_Silicones_Yes_but_only_roughly_and_you_must_choose_your_modulus_carefully
        youngs = 0.0037*np.exp(0.0718 * shore) # MPa
        
        youngs = youngs * 1e6 # Pa
        youngs = np.round(youngs)
        return youngs

    def prepare_data(self):
        print("Start generating data")
        dataset = None
        folds = None
        if not self.path_to_stored_data.exists():
            path_to_data = self.data_dir / self.folder_name

            datapoint_shape = []
            datapoint_shore = []
            datapoint_object = []
            datapoint_id = []
            datapoint_frames = []
            for datapoint_path in tqdm.tqdm([tmp for tmp in path_to_data.iterdir()]):
                infos = datapoint_path.stem.split("_")

                shore = int(infos[1])

                datapoint_shape.append(infos[0])
                datapoint_shore.append(shore)
                datapoint_id.append(int(infos[2]))
                datapoint_object.append(f"{infos[0]}_{infos[1]}") # Object only identifiable through shape and hardness
                
                video = self.extract_frames_from_video(datapoint_path)
                # print(video.shape)
                datapoint_frames.append(video)
                #print(infos)
                # print(datapoint_path)
            
            datapoint_shape = np.array(datapoint_shape)
            datapoint_shore = np.array(datapoint_shore)
            datapoint_youngs = self.normalization_function(self.shore_00_to_youngs(datapoint_shore))
            datapoint_id = np.array(datapoint_id)
            datapoint_frames = np.array(datapoint_frames)
            datapoint_object = np.array(datapoint_object)
            # print(datapoint_frames.shape)

            dataset = {
                "shape": datapoint_shape,
                "shore": datapoint_shore,
                "youngs": datapoint_youngs,
                "id": datapoint_id,
                "frames": datapoint_frames,
                "object": datapoint_object
            }

            with self.path_to_stored_data.open("wb") as f:
                pickle.dump(dataset, f)

        else:
            print("Already Exists")
            if not self.path_to_folds.exists():
                print("Loading data to allow fold generation")
                with self.path_to_stored_data.open("rb") as f:
                    dataset = pickle.load(f)
        
        print("Generated all data")
        print("Start generating folds")
        if not self.path_to_folds.exists():
            folds = {}

            all_indices = np.arange(len(dataset["shape"]))

            r_split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state) # hardcoded seed to always ensure same test set
            
            if self.use_cross_validation:
                k_fold_split = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            else:
                k_fold_split = ShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=self.random_state)

            if self.seen:
                if self.same_test_set:
                    train_indices, test_indices = next(r_split.split(all_indices))

                    for i, (train_split, val_split) in enumerate(k_fold_split.split(all_indices[train_indices])):
                        folds[i] = {
                            "train": train_indices[train_split],
                            "val": train_indices[val_split],
                            "test": test_indices # Always the same test set
                        }
                else:
                    for i in range(self.n_splits):
                        r_split = ShuffleSplit(1, test_size=0.2, random_state=self.random_state+i)
                        train_indices, test_indices = next(r_split.split(all_indices))
                        train_split, val_split = next(r_split.split(all_indices[train_indices]))
                        folds[i] = {
                            "train": train_indices[train_split],
                            "val": train_indices[val_split],
                            "test": test_indices # Always the same test set
                        }
            else:
                all_objects = np.array(list(set(dataset["object"])))
                if self.same_test_set:    
                    train_object_indices, test_object_indices = next(r_split.split(all_objects))
                    print(train_object_indices)
                    print(all_objects)
                    train_objects = all_objects[train_object_indices]
                    test_objects = all_objects[test_object_indices]

                    test_indices = all_indices[np.isin(dataset["object"], test_objects)]
                    train_indices = all_indices[np.isin(dataset["object"], train_objects)]

                    for i, (train_split, val_split) in enumerate(k_fold_split.split(train_objects)):

                        train_data= train_indices[np.isin(dataset["object"][train_indices], train_objects[train_split])]
                        val_data = train_indices[np.isin(dataset["object"][train_indices], train_objects[val_split])]

                        folds[i] = {
                            "train": train_data,
                            "val": val_data,
                            "test": test_indices # Always the same test set
                        }
                else:
                    for i in range(self.n_splits):
                        r_split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state + i)
                        train_object_indices, test_object_indices = next(r_split.split(all_objects))

                        train_objects = all_objects[train_object_indices]
                        test_objects = all_objects[test_object_indices]

                        test_indices = all_indices[np.isin(dataset["object"], test_objects)]
                        train_indices = all_indices[np.isin(dataset["object"], train_objects)]

                        train_split, val_split = next(r_split.split(train_objects))

                        train_data= train_indices[np.isin(dataset["object"][train_indices], train_objects[train_split])]
                        val_data = train_indices[np.isin(dataset["object"][train_indices], train_objects[val_split])]
                        
                        folds[i] = {
                            "train": train_data,
                            "val": val_data,
                            "test": test_indices # Always the same test set
                        }
            
            with self.path_to_folds.open("wb") as f:
                pickle.dump(folds, f)
        else:
            print("Already Exists")
        del dataset
        del folds
        print("Generated all folds")

    def setup(self, stage=None, fold=0):
        
        with self.path_to_stored_data.open("rb") as f:
            dataset = pickle.load(f)

        
        with self.path_to_folds.open("rb") as f:
            folds = pickle.load(f)
        
        print(f"Train Size: {len(folds[fold]['train'])}\nVal Size: {len(folds[fold]['val'])}\nTest Size: {len(folds[fold]['test'])}")

        self.data = {
            "full":{
                "x_frames": dataset["frames"],
                "shore": dataset["shore"],
                "object_name": dataset["shape"],
                "y_label": dataset["youngs"],
                "id": dataset["id"]
            },
            "train":{
                "x_frames": dataset["frames"][folds[fold]["train"]],
                "shore": dataset["shore"][folds[fold]["train"]],
                "object_name": dataset["shape"][folds[fold]["train"]],
                "y_label": dataset["youngs"][folds[fold]["train"]],
                "id": dataset["id"][folds[fold]["train"]]
            },
            "val":{
                "x_frames": dataset["frames"][folds[fold]["val"]],
                "shore": dataset["shore"][folds[fold]["val"]],
                "object_name": dataset["shape"][folds[fold]["val"]],
                "y_label": dataset["youngs"][folds[fold]["val"]],
                "id": dataset["id"][folds[fold]["val"]]
            },
            "test":{
                "x_frames": dataset["frames"][folds[fold]["test"]],
                "shore": dataset["shore"][folds[fold]["test"]],
                "object_name": dataset["shape"][folds[fold]["test"]],
                "y_label": dataset["youngs"][folds[fold]["test"]],
                "id": dataset["id"][folds[fold]["test"]]
            }

        }

    def train_dataloader(self):
        return DataLoader(ShoreEstimatedYoungsDataset(self.data["train"]), batch_size=self.batch_size, shuffle=True, num_workers=self.worker)

    def val_dataloader(self):
        return DataLoader(ShoreEstimatedYoungsDataset(self.data["val"]), batch_size=self.batch_size, shuffle=False, num_workers=self.worker)

    def test_dataloader(self):
        return DataLoader(ShoreEstimatedYoungsDataset(self.data["test"]), batch_size=self.batch_size, shuffle=False, num_workers=self.worker)
    
    def full_dataloader(self):
        return DataLoader(ShoreEstimatedYoungsDataset(self.data["full"]), batch_size=self.batch_size, shuffle=False, num_workers=self.worker)


def main():
    data_dir = "/home/malte.kuhlmann/youngs-modulus/data"
    folder_name = "hardness_ds"

    data_module = KFoldShoreDataModule(data_dir, folder_name, seen=False)
    data_module.prepare_data()
    data_module.setup(fold=8)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    for batch in train_loader:
        x_frames, force, width, estimation, y, object_name = batch
        print(x_frames)
        print(y)
        break

if __name__ == "__main__":
    main()
