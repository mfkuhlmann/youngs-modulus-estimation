import pickle

import hydra
from model.VGG_LSTM.modulus_model import ModulusModel as VGG_LSTM
from model.Res_TF.modulus_model import ModulusModel as Res_TF
from model.Top10NN.modulus_model import ModulusModel as Top10NN
from dataset.k_fold_youngs_data_module import KFoldYoungsDataModule
import tqdm
import torch
from sklearn.metrics import r2_score
import gc
import numpy as np

model_class_dict = {
    "vgg_lstm": VGG_LSTM,
    "res_tf": Res_TF,
    "top10nn": Top10NN,
}

def dict_to_importable_string(res_dict, stat_name="n_mse_loss"):
    importable_strings_dict = {}
    for key in res_dict.keys():
        importable_strings_dict[key] = {}
        importable_string = ""
        for model_name in res_dict[key].keys():
            importable_string += f"{model_name}"
            for fold in res_dict[key][model_name].keys():
                if fold == "mean" or fold == "std":
                    continue
                importable_string += f" {res_dict[key][model_name][fold][stat_name]}"
            importable_string += f" {res_dict[key][model_name]['mean']} {res_dict[key][model_name]['std']}"
            importable_strings_dict[key][model_name] = importable_string
        
    return importable_strings_dict

def evaluate_model(model_name, checkpoint, data_module):
    
    fold = int(checkpoint.parent.stem.split("_")[-1])
    
    
    model = model_class_dict[model_name].load_from_checkpoint(checkpoint)
    model.eval()
    model.hparams.use_transformations = False # To make evaluation deterministic
    model_config = model.config
    
    data = data_module.test_dataloader()

    stats = {
        "loss": [],
        "abs_log_diff": [],
        "log_accuracy": [],
        "shape_log_accuracy": {},
        "shape_n_mse": {},
        "shape_std": {},
        "shape_correct": {},
        "shape_count": {},
        "shape_r2_score": {},
        "material_log_accuracy": {},
        "material_n_mse": {},
        "material_std": {},
        "material_correct": {},
        "material_count": {},
        "material_r2_score": {},
    }
    shape_preds = {}
    shape_targets = {}
    material_preds = {}
    material_targets = {}
    for shape in set(data_module.object_to_shape.values()):
        stats["shape_log_accuracy"][shape] = []
        stats["shape_n_mse"][shape] = []
        shape_preds[shape] = []
        shape_targets[shape] = []
    for material in set(data_module.object_to_material.values()):
        stats["material_log_accuracy"][material] = []
        stats["material_n_mse"][material] = []
        material_preds[material] = []
        material_targets[material] = []

    
    all_preds = []
    all_targets = []
    for batch in tqdm.tqdm(data):
        x_frames, x_forces, x_widths, x_estimations, y, object_name = batch
        batch_size = x_frames.shape[0]
        y = y.squeeze()
        outputs = model.forward(x_frames.cuda(), x_forces.cuda(), x_widths.cuda(), x_estimations.cuda(), x_frames.shape[0])
        outputs = outputs.squeeze()
        loss = torch.nn.functional.mse_loss(outputs, y.cuda()).cpu()
        abs_log_diff = torch.abs(torch.log10(model.unnormalization_function(outputs.cpu())) - torch.log10(model.unnormalization_function(y.cpu()))).detach().numpy()

        all_preds.append(outputs.cpu().detach().numpy())
        all_targets.append(y.cpu().detach().numpy())

        shape_preds[data_module.object_to_shape[object_name[0]]].append(outputs.cpu().detach().numpy())
        shape_targets[data_module.object_to_shape[object_name[0]]].append(y.cpu().detach().numpy())
        material_preds[data_module.object_to_material[object_name[0]]].append(outputs.cpu().detach().numpy())
        material_targets[data_module.object_to_material[object_name[0]]].append(y.cpu().detach().numpy())

        stats["log_accuracy"].append(1 if (abs_log_diff <= 1) else 0)
        stats["loss"].append(loss.detach().numpy())
        stats["abs_log_diff"].append(abs_log_diff)

        stats["shape_log_accuracy"][data_module.object_to_shape[object_name[0]]].append(1 if (abs_log_diff <= 1) else 0)
        stats["material_log_accuracy"][data_module.object_to_material[object_name[0]]].append(1 if (abs_log_diff <= 1) else 0)

        stats["shape_n_mse"][data_module.object_to_shape[object_name[0]]].append(loss.detach().numpy())
        stats["material_n_mse"][data_module.object_to_material[object_name[0]]].append(loss.detach().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    stats["loss"] = np.array(stats["loss"])
    stats["r2_score"] = r2_score(all_targets, all_preds)
    stats["n_mse_loss"] = np.mean(stats["loss"])
    stats["abs_log_diff"] = np.mean(stats["abs_log_diff"])
    stats["log_accuracy"] = np.mean(stats["log_accuracy"])
    stats["n_mse_std"] = np.std(stats["loss"])

    for shape in set(data_module.object_to_shape.values()):
        stats["shape_count"][shape] = len(stats["shape_log_accuracy"][shape])
        stats["shape_correct"][shape] = sum(stats["shape_log_accuracy"][shape])
        stats["shape_log_accuracy"][shape] = np.mean(stats["shape_log_accuracy"][shape]) if len(stats["shape_log_accuracy"][shape]) > 0 else -1
        stats["shape_std"][shape] = np.std(stats["shape_n_mse"][shape]) if len(stats["shape_n_mse"][shape]) > 0 else -1
        stats["shape_r2_score"][shape] = r2_score(shape_targets[shape], shape_preds[shape]) if len(stats["shape_n_mse"][shape]) > 0 else -1
        stats["shape_n_mse"][shape] = np.mean(stats["shape_n_mse"][shape]) if len(stats["shape_n_mse"][shape]) > 0 else -1

    for material in set(data_module.object_to_material.values()):
        stats["material_count"][material] = len(stats["material_log_accuracy"][material])
        stats["material_correct"][material] = sum(stats["material_log_accuracy"][material])
        stats["material_log_accuracy"][material] = np.mean(stats["material_log_accuracy"][material]) if len(stats["material_log_accuracy"][material]) > 0 else -1
        stats["material_std"][material] = np.std(stats["material_n_mse"][material]) if len(stats["material_n_mse"][material]) > 0 else -1
        stats["material_r2_score"][material] = r2_score(material_targets[material], material_preds[material]) if len(stats["material_n_mse"][material]) > 0 else -1
        stats["material_n_mse"][material] = np.mean(stats["material_n_mse"][material]) if len(stats["material_n_mse"][material]) > 0 else -1

    return stats

from pathlib import Path

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg):
    path_to_checkpoints = Path(cfg['path_to_checkpoints']).resolve()

    results = {}

    result_path = Path(cfg['evaluation_result_path']).resolve()
    result_path.mkdir(exist_ok=True)

    for model_folder in path_to_checkpoints.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.stem
        if cfg['model_name'] is not None:
            if cfg['model_name'] != model_name:
                continue

        if (result_path / model_name).exists():
            continue
        else:
            (result_path / model_name).mkdir(exist_ok=True)


        fold_0_folder = model_folder / "fold_0"
        print(fold_0_folder)
        for checkpoint in fold_0_folder.iterdir():
            checkpoint_name = checkpoint.name
            active_dataset = checkpoint.stem.split("_", 1)[-1]
            results[active_dataset] = {}
            results[active_dataset][model_name] = {}

            m = model_class_dict[model_name].load_from_checkpoint(checkpoint)
            model_config = m.config
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
            )
            data_module.prepare_data()
            # data_module.setup()
            data_module.make_object_attributes()
            
            for fold in model_folder.iterdir():
                checkpoint = fold / checkpoint_name
                if not checkpoint.exists():
                    continue
                
                print(checkpoint)
                fold = int(fold.stem.split("_")[-1])
                data_module.setup(fold=fold)

                result_dict = evaluate_model(model_name, checkpoint, data_module=data_module)
                results[active_dataset][model_name][fold] = result_dict
                print(result_dict)

                (result_path / model_name / f"{fold}").mkdir(exist_ok=True)
                with (result_path / model_name / f"{fold}" / f"{active_dataset}.pkl").open("wb") as f:
                    pickle.dump(result_dict, f)

            results[active_dataset][model_name]["mean"] = np.mean([results[active_dataset][model_name][fold]["n_mse_loss"] for fold in results[active_dataset][model_name] if fold != "mean"])
            results[active_dataset][model_name]["std"] = np.std([results[active_dataset][model_name][fold]["n_mse_loss"] for fold in results[active_dataset][model_name] if fold != "mean"])
            
            del data_module
            gc.collect()

    with (result_path / "results.pkl").open("wb") as f:
        pickle.dump(results, f)

    importable_strings_dict = dict_to_importable_string(results)
    for key in importable_strings_dict.keys():
        print(f"{key}")
        print("Model Name, Fold 0, Fold 1, Fold 2, Fold 3, Fold 4, Fold 5, Fold 6, Fold 7, Fold 8, Fold 9, Mean, Std")
        for model_name in importable_strings_dict[key].keys():
            print(f"{importable_strings_dict[key][model_name].replace('.', ',')}")

if __name__ == "__main__":
    main()