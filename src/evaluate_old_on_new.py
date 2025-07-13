import pickle
from dataset.k_fold_shore_data_module import KFoldShoreDataModule
from model.VGG_LSTM.modulus_model import ModulusModel as VGG_LSTM
from model.Res_TF.modulus_model import ModulusModel as Res_TF
from model.Top10NN.modulus_model import ModulusModel as Top10NN
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
    
    data = data_module.full_dataloader()

    stats = {
        "loss": [],
        "abs_log_diff": [],
        "log_accuracy": [],
    }


    
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
        #print(outputs)
        #print(y)
        #print(abs_log_diff)
        all_preds.append(outputs.cpu().detach().numpy())
        all_targets.append(y.cpu().detach().numpy())

        stats["log_accuracy"].append(1 if (abs_log_diff <= 1) else 0)
        stats["loss"].append(loss.detach().numpy())
        stats["abs_log_diff"].append(abs_log_diff)

    all_preds = np.array(all_preds)
    print(all_preds)
    all_targets = np.array(all_targets)
    stats["loss"] = np.array(stats["loss"])
    stats["r2_score"] = r2_score(all_targets, all_preds)
    stats["n_mse_loss"] = np.mean(stats["loss"])
    stats["abs_log_diff"] = np.mean(stats["abs_log_diff"])
    stats["log_accuracy"] = np.mean(stats["log_accuracy"])
    stats["n_mse_std"] = np.std(stats["loss"])

    return stats

from pathlib import Path

path_to_checkpoints = Path("../checkpoint_balanced_not_same_test_set").resolve()

results = {}

result_path = Path("../data/results_old_on_new_higher_norm_diff").resolve()
result_path.mkdir(exist_ok=True)

for model_folder in path_to_checkpoints.iterdir():
    if not model_folder.is_dir():
        continue

    model_name = model_folder.stem

    if (result_path / model_name).exists():
        continue
    else:
        (result_path / model_name).mkdir(exist_ok=True)


    fold_0_folder = model_folder / "fold_0"
    print(fold_0_folder)

    data_module= KFoldShoreDataModule(
        data_dir = "/home/malte.kuhlmann/youngs-modulus/data",
        folder_name = "hardness_ds",
        worker = 5,
        seen=True,
        batch_size=1,
        n_splits=10,
        use_cross_validation=False,
        only_evaluation=True,
        random_state=0,
        intensity_threshold=0.5
    )
    data_module.prepare_data()
    data_module.setup()

    for checkpoint in fold_0_folder.iterdir():
        checkpoint_name = checkpoint.name
        active_dataset = checkpoint.stem.split("_", 1)[-1]
        results[active_dataset] = {}
        results[active_dataset][model_name] = {}

        m = model_class_dict[model_name].load_from_checkpoint(checkpoint)
        model_config = m.config

        # data_module.make_object_attributes()
        
        for fold in model_folder.iterdir():
            checkpoint = fold / checkpoint_name
            if not checkpoint.exists():
                continue
            
            print(checkpoint)
            fold = int(fold.stem.split("_")[-1])
            result_dict = evaluate_model(model_name, checkpoint, data_module=data_module)
            results[active_dataset][model_name][fold] = result_dict
            print(result_dict)

            (result_path / model_name / f"{fold}").mkdir(exist_ok=True)
            with (result_path / model_name / f"{fold}" / f"{active_dataset}.pkl").open("wb") as f:
                pickle.dump(result_dict, f)

        results[active_dataset][model_name]["mean"] = np.mean([results[active_dataset][model_name][fold]["n_mse_loss"] for fold in results[active_dataset][model_name] if fold != "mean"])
        results[active_dataset][model_name]["std"] = np.std([results[active_dataset][model_name][fold]["n_mse_loss"] for fold in results[active_dataset][model_name] if fold != "mean"])


with (result_path / "old_to_new_results.pkl").open("wb") as f:
    pickle.dump(results, f)

importable_strings_dict = dict_to_importable_string(results)
for key in importable_strings_dict.keys():
    print(f"{key}")
    print("Model Name, Fold 0, Fold 1, Fold 2, Fold 3, Fold 4, Fold 5, Fold 6, Fold 7, Fold 8, Fold 9, Mean, Std")
    for model_name in importable_strings_dict[key].keys():
        print(f"{importable_strings_dict[key][model_name].replace('.', ',')}")