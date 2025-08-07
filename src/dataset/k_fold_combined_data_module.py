from collections import Counter
import csv
import os
from pathlib import Path
import pickle
import lightning
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import StratifiedKFold
import torchvision
import h5py
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import argparse


class CombinedDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self):
        return len(self.data["x_frames"])
    
    def __getitem__(self, idx):
        return self.data["x_frames"][idx], self.data["x_forces"][idx], self.data["x_widths"][idx], self.data["x_estimations"][idx], self.data["y_label"][idx], self.data["object_name"][idx]



class KFoldCombinedDataModule(lightning.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, n_splits=5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.config = {
            "use_width_transforms": True
        }

        # TODO: generate dataset names
        self.modulus_data = data_dir / "dataset_variants" / "dataset_shape_material_unseen_width_transforms_log_normalize_removed_paper_random_state_42_shape_material_dataset.pkl"
        self.shore_data = data_dir / "dataset_variants" / "shore_full_data.pk"
        self.shore_folds = data_dir / "k_fold_shore_seen_data_folds.pk"

    def prepare_data(self):
        # TODO: do asserts
        assert self.modulus_data.exists()
        assert self.shore_data.exists()
        assert self.shore_folds.exists()

    def setup(self, stage=None, fold=0):
        assert fold < self.n_splits # Making sure that the fold actually exists 

        youngs_dataset = pickle.load(self.modulus_data.open('rb'))

        with self.shore_data.open("rb") as f:
            dataset = pickle.load(f)

        
        with self.shore_folds.open("rb") as f:
            folds = pickle.load(f)

        shore_data = {
            "train":{
                "x_frames": torch.movedim(torch.tensor(dataset["frames"][folds[fold]["train"]]), 4, 2),
                "shore": dataset["shore"][folds[fold]["train"]],
                "object_name": dataset["shape"][folds[fold]["train"]],
                "y_label": torch.unsqueeze(torch.unsqueeze(torch.tensor(dataset["youngs"][folds[fold]["train"]]), -1), -1),
                "id": dataset["id"][folds[fold]["train"]]
            },
            "val":{
                "x_frames": torch.movedim(torch.tensor(dataset["frames"][folds[fold]["val"]]), 4, 2),
                "shore": dataset["shore"][folds[fold]["val"]],
                "object_name": dataset["shape"][folds[fold]["val"]],
                "y_label": torch.unsqueeze(torch.unsqueeze(torch.tensor(dataset["youngs"][folds[fold]["val"]]), -1), -1),
                "id": dataset["id"][folds[fold]["val"]]
            },
            "test":{
                "x_frames": torch.movedim(torch.tensor(dataset["frames"][folds[fold]["test"]]), 4, 2),
                "shore": dataset["shore"][folds[fold]["test"]],
                "object_name": dataset["shape"][folds[fold]["test"]],
                "y_label": torch.unsqueeze(torch.unsqueeze(torch.tensor(dataset["youngs"][folds[fold]["test"]]), -1), -1),
                "id": dataset["id"][folds[fold]["test"]]
            }
            }

        # TODO: test set with data augmentation?
        if self.config["use_width_transforms"]:
            self.train_indices = torch.cat([torch.tensor(youngs_dataset["splits"][fold]["train"]), torch.tensor(np.array(youngs_dataset["splits"][fold]["train"]) + len(youngs_dataset["all_data"]["x_frames"])//2)])
            self.val_indices = torch.cat([torch.tensor(youngs_dataset["splits"][fold]["val"]), torch.tensor(np.array(youngs_dataset["splits"][fold]["val"]) + len(youngs_dataset["all_data"]["x_frames"])//2)])
            self.test_indices = torch.cat([torch.tensor(youngs_dataset["splits"][fold]["test"]), torch.tensor(np.array(youngs_dataset["splits"][fold]["test"]) + len(youngs_dataset["all_data"]["x_frames"])//2)])
            


            print(self.train_indices.tolist())
            print(self.val_indices.tolist())
            print(self.test_indices.tolist())

            #assert torch.all(train_indices != val_indices)
            #assert torch.all(val_indices != test_indices)
            #assert torch.all(train_indices != test_indices)

            print(f"Train: {len(self.train_indices)}\nVal: {len(self.val_indices)}\nTest: {len(self.test_indices)}")


            print(youngs_dataset["all_data"]["y_label"][self.test_indices].size())
            print(shore_data["test"]["y_label"].size())
            self.dataset = {}
            self.dataset["train"] = {
                "x_frames": torch.cat([youngs_dataset["all_data"]["x_frames"][self.train_indices], shore_data["train"]["x_frames"]]),
                "x_forces": torch.cat([youngs_dataset["all_data"]["x_forces"][self.train_indices], torch.zeros((len(shore_data["train"]["x_frames"]), 3, 1))]),
                "x_widths": torch.cat([youngs_dataset["all_data"]["x_widths"][self.train_indices], torch.zeros((len(shore_data["train"]["x_frames"]), 3, 1))]),
                "x_estimations": torch.cat([youngs_dataset["all_data"]["x_estimations"][self.train_indices], torch.zeros((len(shore_data["train"]["x_frames"]), 2, 1))]),
                "y_label": torch.cat([youngs_dataset["all_data"]["y_label"][self.train_indices], shore_data["train"]["y_label"]]),
                "object_name": [youngs_dataset["all_data"]["object_name"][o] for o in youngs_dataset["splits"][fold]["train"]] + [youngs_dataset["all_data"]["object_name"][len(youngs_dataset["all_data"]["object_name"])//2:][o] for o in youngs_dataset["splits"][fold]["train"]] + shore_data["train"]["object_name"].tolist()
            }
            self.dataset["val"] = {
                "x_frames": torch.cat([youngs_dataset["all_data"]["x_frames"][self.val_indices], shore_data["val"]["x_frames"]]),
                "x_forces": torch.cat([youngs_dataset["all_data"]["x_forces"][self.val_indices], torch.zeros((len(shore_data["val"]["x_frames"]), 3, 1))]),
                "x_widths": torch.cat([youngs_dataset["all_data"]["x_widths"][self.val_indices], torch.zeros((len(shore_data["val"]["x_frames"]), 3, 1))]),
                "x_estimations": torch.cat([youngs_dataset["all_data"]["x_estimations"][self.val_indices], torch.zeros((len(shore_data["val"]["x_frames"]), 2, 1))]),
                "y_label": torch.cat([youngs_dataset["all_data"]["y_label"][self.val_indices], shore_data["val"]["y_label"]]),
                "object_name": [youngs_dataset["all_data"]["object_name"][o] for o in youngs_dataset["splits"][fold]["val"]] + [youngs_dataset["all_data"]["object_name"][len(youngs_dataset["all_data"]["object_name"])//2:][o] for o in youngs_dataset["splits"][fold]["val"]] + shore_data["val"]["object_name"].tolist()
            }
            self.dataset["test"] = {
                "x_frames": torch.cat([youngs_dataset["all_data"]["x_frames"][self.test_indices], shore_data["test"]["x_frames"]]),
                "x_forces": torch.cat([youngs_dataset["all_data"]["x_forces"][self.test_indices], torch.zeros((len(shore_data["test"]["x_frames"]), 3, 1))]),
                "x_widths": torch.cat([youngs_dataset["all_data"]["x_widths"][self.test_indices], torch.zeros((len(shore_data["test"]["x_frames"]), 3, 1))]),
                "x_estimations": torch.cat([youngs_dataset["all_data"]["x_estimations"][self.test_indices], torch.zeros((len(shore_data["val"]["x_frames"]), 2, 1))]),
                "y_label": torch.cat([youngs_dataset["all_data"]["y_label"][self.test_indices], shore_data["test"]["y_label"]]),
                "object_name": [youngs_dataset["all_data"]["object_name"][o] for o in youngs_dataset["splits"][fold]["test"]] + [youngs_dataset["all_data"]["object_name"][len(youngs_dataset["all_data"]["object_name"])//2:][o] for o in youngs_dataset["splits"][fold]["test"]] + shore_data["test"]["object_name"].tolist()
            }

        else:
            self.train_indices = youngs_dataset["splits"][fold]["train"]
            self.val_indices = youngs_dataset["splits"][fold]["val"]
            self.test_indices = youngs_dataset["splits"][fold]["test"]

            #print(train_indices)
            #print(val_indices)
            #print(test_indices)

            print(f"Train: {len(self.train_indices)}\nVal: {len(self.val_indices)}\nTest: {len(self.test_indices)}")

            assert torch.all(torch.tensor(self.train_indices) != torch.tensor(self.val_indices))
            assert torch.all(torch.tensor(self.val_indices) != torch.tensor(self.test_indices))
            assert torch.all(torch.tensor(self.train_indices) != torch.tensor(self.test_indices))


            self.dataset = {
                "train": {
                    "x_frames": torch.cat([youngs_dataset["all_data"]["x_frames"][self.train_indices].detach(), shore_data["train"]["x_frames"].detach()]),
                    "x_forces": torch.cat([self.dataset["all_data"]["x_forces"][self.train_indices].detach(), torch.zeros((len(shore_data["train"]["x_frames"]), 3)).detach()]),
                    "x_widths": torch.cat([youngs_dataset["all_data"]["x_widths"][self.train_indices].detach(), torch.zeros((len(shore_data["train"]["x_frames"]), 3)).detach()]),
                    "x_estimations": torch.cat([youngs_dataset["all_data"]["x_estimations"][self.train_indices].detach(), torch.zeros((len(shore_data["train"]["x_frames"]), 2)).detach()]),
                    "y_label": torch.cat([youngs_dataset["all_data"]["y_label"][self.train_indices].detach(), shore_data["train"]["y_label"].detach()]),
                    "object_name": [youngs_dataset["all_data"]["object_name"][o] for o in youngs_dataset["splits"][fold]["train"]] + shore_data["train"]["object_name"].tolist()
                },
                "val": {
                    "x_frames": torch.cat([youngs_dataset["all_data"]["x_frames"][self.val_indices].detach(), shore_data["val"]["x_frames"].detach()]),
                    "x_forces": torch.cat([youngs_dataset["all_data"]["x_forces"][self.val_indices].detach(), torch.zeros((len(shore_data["val"]["x_frames"]), 3)).detach()]),
                    "x_widths": torch.cat([youngs_dataset["all_data"]["x_widths"][self.val_indices].detach(), torch.zeros((len(shore_data["val"]["x_frames"]), 3)).detach()]),
                    "x_estimations": torch.cat([youngs_dataset["all_data"]["x_estimations"][self.val_indices].detach(), torch.zeros((len(shore_data["val"]["x_frames"]), 2)).detach()]),
                    "y_label": torch.cat([youngs_dataset["all_data"]["y_label"][self.val_indices].detach(), shore_data["val"]["y_label"].detach()]),
                    "object_name": [youngs_dataset["all_data"]["object_name"][o] for o in youngs_dataset["splits"][fold]["val"]] + shore_data["val"]["object_name"].tolist()
                },
                "test": {
                    "x_frames": torch.cat([youngs_dataset["all_data"]["x_frames"][self.test_indices].detach(), shore_data["test"]["x_frames"].detach()]),
                    "x_forces": torch.cat([youngs_dataset["all_data"]["x_forces"][self.test_indices].detach(), torch.zeros((len(shore_data["test"]["x_frames"]), 3)).detach()]),
                    "x_widths": torch.cat([youngs_dataset["all_data"]["x_widths"][self.test_indices].detach(), torch.zeros((len(shore_data["test"]["x_frames"]), 3)).detach()]),
                    "x_estimations": torch.cat([youngs_dataset["all_data"]["x_estimations"][self.test_indices].detach(), torch.zeros((len(shore_data["val"]["x_frames"]), 2)).detach()]),
                    "y_label": torch.cat([youngs_dataset["all_data"]["y_label"][self.test_indices].detach(), shore_data["test"]["y_label"].detach()]),
                    "object_name": [youngs_dataset["all_data"]["object_name"][o] for o in youngs_dataset["splits"][fold]["test"]] + shore_data["test"]["object_name"].tolist()
                },
            }

        # self.dataset = torch.load(self.full_dataset_file)

        return super().setup(stage)

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        return DataLoader(CombinedDataset(self.dataset["train"]), batch_size=batch_size, shuffle=True, num_workers=self.worker)
    
    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(CombinedDataset(self.dataset["val"]), batch_size=batch_size, shuffle=False, num_workers=self.worker)
    
    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(CombinedDataset(self.dataset["test"]), batch_size=batch_size, shuffle=False, num_workers=self.worker)
    
if __name__ == "__main__":
    data_dir = Path("/home/malte.kuhlmann/youngs-modulus/data")
    batch_size = 32
    num_workers = 4
    n_splits = 5
    fold = 0

    data_module = KFoldCombinedDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        n_splits=n_splits
    )

    data_module.prepare_data()
    data_module.setup(fold=fold)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Train loader length: {len(train_loader)}")
    print(f"Validation loader length: {len(val_loader)}")
    print(f"Test loader length: {len(test_loader)}")
