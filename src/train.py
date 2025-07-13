import json
from pathlib import Path
import random
import hydra
import numpy as np
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

    name += f"_random_state_{42}"
    return name

def get_best_balancing_threshold(cfg):
    path = Path(cfg["settings"]["threshold_smac_optimization_dir"]) / "0" / "runhistory.json"
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

def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed != -1:
        # the seed is the jobid which has . in it
        #seed = seed.replace(".", "")
        #seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        lightning.seed_everything(seed, workers=True)

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        g = torch.Generator()
        g.manual_seed(0)
        return g
    else:
        print("Not setting seed.")
        return None


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    


    model_config = {}
    print(cfg["dataset"]["model"][cfg["settings"]["model_name"]])

    if cfg["settings"]["use_optimized_architecture"]:
        # Get best architecture from smac_optimization
        best_architecture = get_best_model_from_smac(cfg)
        print("Best Model Architecture:")
        print(best_architecture)
        c = cfg["dataset"]["model"][cfg["settings"]["model_name"]].copy()
        c.update(best_architecture)

        model_config.update(c)
    else:
        model_config.update(cfg["dataset"]["model"][cfg["settings"]["model_name"]])
    
    

    cfg["dataset"].pop("model") 
    model_config.update(cfg["dataset"])
    model_config['random_state'] = cfg["settings"]["random_state"]

    if cfg["settings"]["use_optimized_balancing_threshold"]:
        best_threshold = get_best_balancing_threshold(cfg)
        print("Best Balancing Threshold:")
        print(best_threshold)
        model_config.update(best_threshold)

    print("Full Model Config:")
    print(model_config)

    set_seed(cfg["settings"]["random_state"])

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
        same_test_set=model_config['same_test_set'],
        use_cross_validation=model_config['use_cross_validation'],
        compliance=model_config['compliance'],
        # exclude_shape=model_config['exclude_shape'],
        random_state=model_config['random_state'],
        balance_dataset=model_config["balance_dataset"],
        balance_position=model_config["balance_position"],
        balance_bucket=model_config["balance_bucket"],
        balance_threshold=model_config["balance_threshold"],
        balance_validation_set=model_config["balance_validation_set"],
        balance_test_set=model_config["balance_test_set"],
        full_dataset_undersample=model_config["full_dataset_undersample"],
        full_dataset_undersample_amount=model_config["full_dataset_undersample_amount"],
    )

    model_config["dataset_name"] = data_module.get_dataset_name() # Add dataset name to model for later grouping
    model_config["model_name"] = cfg["settings"]["model_name"] # Add model name to model for later grouping

    # quit()
    if cfg["settings"]["model_name"] == "res_tf":
        model = Res_TFModulusModel(model_config)
        print("Using Res_TF model")
    elif cfg["settings"]["model_name"] == "top10nn":
        model = Top10NNModulusModel(model_config)
        print("Using Top10NN model")
    elif cfg["settings"]["model_name"] == "vgg_lstm":
        model = VGG_LSTMModulusModel(model_config)
        print("Using VGG_LSTM model")
    else:
        raise ValueError(f"Model name {cfg['settings']['model_name']} not recognized")
    
    data_module.prepare_data()
    data_module.setup(fold=cfg["settings"]["fold"])

    run_name = cfg["settings"]["model_name"]

    if "W_API_KEY" in os.environ:
        wandb.login(key=os.environ["W_API_KEY"])

    logger = WandbLogger(
        project=cfg["logger"]["project"],
        save_dir=cfg["logger"]["save_dir"],
        name=f"{run_name}_{data_module.get_dataset_name()[8:-12]}_{cfg['settings']['fold']}",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg["settings"]["checkpoint_dir"]).resolve() / f"{run_name}" / f"fold_{cfg['settings']['fold']}",
        save_top_k=1,
        verbose=True,
        monitor='val_n_mse',
        mode='min',
        filename=f"{run_name}_{data_module.get_dataset_name()[8:-12]}"
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"{run_name}_{data_module.get_dataset_name()[8:-12]}_last"

    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    early_stopping = EarlyStopping(monitor="val_n_mse", mode="min", patience=30)
    # torch.set_float32_matmul_precision("medium")

    cbacks = []
    if cfg["settings"]["use_checkpointing"]:
        cbacks.append(checkpoint_callback)
    
    if cfg["settings"]["use_early_stopping"]:
        cbacks.append(early_stopping)

    trainer = lightning.Trainer(
        max_epochs=cfg["settings"]["max_epochs"],
        devices=1,
        logger=logger,
        callbacks=cbacks,
        precision=cfg["settings"]["precision"],
        # check_val_every_n_epoch=20
    )

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
    trainer.test(model, dataloaders=data_module.test_dataloader()) 
    
    del data_module
    gc.collect()
    wandb.finish()


if __name__ == "__main__":
    main()