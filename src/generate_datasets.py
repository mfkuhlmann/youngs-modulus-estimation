import json
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import lightning
import torch
import os
import gc
import wandb
from dataset.k_fold_youngs_data_module import KFoldYoungsDataModule

from model.Res_TF.modulus_model import ModulusModel as Res_TFModulusModel
from model.Top10NN.modulus_model import ModulusModel as Top10NNModulusModel
from model.VGG_LSTM.modulus_model import ModulusModel as VGG_LSTMModulusModel

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def get_name(model_name, cfg):
    name = model_name

    if cfg["rubber_only"]:
        name += "_rubber_only"

    name += f"_{cfg['sample_type']}"
    if cfg["val_on_seen_objects"]:
        name += "_seen"
    else:
        name += "_unseen"
    
    if cfg["use_estimations"]:
        name += "_est"
    
    if cfg["use_force"]:
        name += "_force"
    
    if cfg["use_width"]:
        name += "_width"
    
    if cfg["use_width_transforms"]:
        name += "_width_transforms"

    if cfg["use_markers"]:
        name += "_only_markers"

    name += f"_random_state_{42}"
    return name


def get_best_model_from_smac(cfg):
    
    translation_dict = {
        "res_tf": "Res_TF",
        "top10nn": "Top10NN",
        "vgg_lstm": "VGG_LSTM"
    }
    optimization_folder_name = get_name(translation_dict[cfg["settings"]["model_name"]], cfg["dataset"])

    path = Path(cfg["settings"]["smac_optimization_dir"]) / optimization_folder_name / "0" / "runhistory.json"
    path = path.resolve()

    with path.open("r") as runhistory_file:
        runhistory = json.load(runhistory_file)

    best_run = None
    for run in runhistory["data"]:
        if best_run is None or run[4] < best_run[4]:
            best_run = run
    
    best_architecture = runhistory['configs'][str(best_run[0])]
    print(f"Best Architecture:\nCost: {best_run[4]}\nConfig: {best_architecture}\n")

    return best_architecture


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    model_config = {}
    print(cfg["dataset"]["model"][cfg["settings"]["model_name"]])

    if cfg["settings"]["use_optimized_architecture"]:
        # Get best architecture from smac_optimization
        best_architecture = get_best_model_from_smac(cfg)
        best_architecture["gamma"] = None
        c = cfg["dataset"]["model"][cfg["settings"]["model_name"]].copy()
        c.update(best_architecture)

        model_config.update(c)
    else:
        model_config.update(cfg["dataset"]["model"][cfg["settings"]["model_name"]])
    
    cfg["dataset"].pop("model") 
    model_config.update(cfg["dataset"])
    print(model_config)

    data_module = KFoldYoungsDataModule(
        data_dir=model_config['data_dir'],
        training_data_folder=model_config['training_data_folder'],
        worker=cfg["settings"]["worker"],
        image_style=model_config['img_style'],
        sample_type=model_config['sample_type'],
        use_estimations=model_config['use_estimations'],
        use_force=model_config['use_force'],
        use_width=model_config['use_width'],
        use_width_transforms=model_config['use_width_transforms'],
        use_markers=model_config['use_markers'],
        remove_paper=model_config['remove_paper'],
        overwrite_file=False,
        batch_size=model_config['batch_size'],
        val_on_seen_objects=model_config["val_on_seen_objects"],
        use_log_normalization=model_config["use_log_normalization"],
        # stratify_with_magnitude=model_config["stratify_with_magnitude"],
        exclude=model_config['exclude'],
        n_splits=model_config['n_splits'],
        use_cross_validation=model_config['use_cross_validation'],
        compliance=model_config['compliance'],
        # exclude_shape=model_config['exclude_shape'],
        random_state=model_config['random_state'],
        balance_dataset=model_config["balance_dataset"],
        balance_position=model_config["balance_position"],
        balance_bucket=model_config["balance_bucket"],
        balance_threshold=model_config["balance_threshold"],
        balance_test_set=model_config["balance_test_set"],
    )

    data_module.prepare_data()
    
    del data_module
    gc.collect()



if __name__ == "__main__":
    main()