import hydra
import lightning

from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from dataset.k_fold_youngs_data_module import KFoldYoungsDataModule
from model.Res_TF.modulus_model import ModulusModel as ResTF
from model.Top10NN.modulus_model import ModulusModel as Top10NN
from model.VGG_LSTM.modulus_model import ModulusModel as VGGLSTM

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac import HyperparameterOptimizationFacade, MultiFidelityFacade, Scenario

class OptimizationMetaData():

    def __init__(self, dataset_cfg, seed=42):
        self.res_tf_model_config = {'gamma': 0.975, 'learning_rate': 0.0001, 'lr_step_size': 1, 'batch_size': 32, 'use_fair_loss': False, 'video_dropout': 0.1, 'force_dropout': 0.1, 'width_dropout': 0.1, 'est_dropout': 0.1, 'decoder_dropout': 0.1, 'fwe_feature_size': 32, 'decoder_size': [512, 512, 128], 'est_decoder_size': [64, 64, 32], 'decoder_output_size': 3, 'layer': 5, 'head': 8, 'embedding_dim': 512, 'output_dim': 512}
        self.res_tf_model_config.update(dataset_cfg)

        self.vgg_lstm_model_config = {'gamma': 0.98, 'learning_rate': 5e-05, 'batch_size': 32, 'lr_step_size': 1, 'lstm_dropout': 0.0, 'input_dim': 4096, 'hidden_dim': 512, 'scalar1_dim': 256, 'scalar2_dim': 8, 'scalar3_dim': 128, 'scalar4_dim': 16}
        self.vgg_lstm_model_config.update(dataset_cfg)

        self.top10nn_model_config = {'gamma': 0.955, 'learning_rate': 7.000000000000001e-05, 'lr_step_size': 1, 'batch_size': 8, 'video_dropout': 0.0, 'force_dropout': 0.3, 'width_dropout': 0.3, 'est_dropout': 0.3, 'decoder_dropout': 0.0, 'img_feature_size': 64, 'fwe_feature_size': 8, 'decoder_size': [512, 512, 256], 'est_decoder_size': [128, 128, 64], 'decoder_output_size': 1, 'random_state': 42}
        self.top10nn_model_config.update(dataset_cfg)
        self.seed = seed

        #model_config = self.res_tf_model_config
        #self.data_module = self.setup_data_module(model_config=model_config)

        
    def setup_data_module(self, model_config):
        data_module = KFoldYoungsDataModule(
            data_dir=model_config['data_dir'],
            training_data_folder=model_config['training_data_folder'],
            worker=10,
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
        data_module.setup()
        return data_module


    @property
    def configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        cs.add([CategoricalHyperparameter("balance_threshold", choices=[0.01 * i for i in range(101)], default_value=0.5),]) # Feature size for force, width. Only one fully connected layer
        
        return cs
    
    def get_name(self):
        name = "Res_TF"

        if self.res_tf_model_config["rubber_only"]:
            name += "_rubber_only"

        name += f"_{self.res_tf_model_config['sample_type']}"
        if self.res_tf_model_config["val_on_seen_objects"]:
            name += "_seen"
        else:
            name += "_unseen"
        
        if self.res_tf_model_config["use_estimations"]:
            name += "_est"
        
        if self.res_tf_model_config["use_force"]:
            name += "_force"
        
        if self.res_tf_model_config["use_width"]:
            name += "_width"
        
        if self.res_tf_model_config["use_width_transforms"]:
            name += "_width_transforms"

        if self.res_tf_model_config["use_markers"]:
            name += "_only_markers"

        name += f"_random_state_{self.seed}"
        return name
    
    def objective_function(self, config: Configuration, seed=0, budget=50):
        
        config = dict(config)

        model_config = self.res_tf_model_config.copy()

        model_config.update(config)
        print(model_config)
        data_module = self.setup_data_module(model_config)

        model_config["dataset_name"] = data_module.get_dataset_name()
        
        res_tf = ResTF(model_config)

        # self.data_module.batch_size = int(model_config['batch_size'])
        
        logger = TensorBoardLogger(save_dir="/home/malte.kuhlmann/youngs-modulus/data/logs", name=data_module.get_dataset_name() + "_res_tf")

        trainer = lightning.Trainer(
            devices=1,
            logger=logger,
            max_epochs=int(budget),
            # precision="bf16-mixed",
            check_val_every_n_epoch=100
        )

        trainer.fit(model=res_tf, train_dataloaders=data_module.train_dataloader(batch_size=int(model_config['batch_size'])), val_dataloaders=data_module.val_dataloader(batch_size=int(model_config['batch_size'])))

        res_tf_validation_metrics = trainer.validate(model=res_tf, dataloaders=data_module.val_dataloader(batch_size=int(model_config['batch_size'])))

        res_tf.cpu()
        del res_tf

        model_config = self.vgg_lstm_model_config.copy()
        model_config.update(config)
        model_config["dataset_name"] = data_module.get_dataset_name()

        vgg_lstm = VGGLSTM(model_config)

        logger = TensorBoardLogger(save_dir="/home/malte.kuhlmann/youngs-modulus/data/logs", name=data_module.get_dataset_name() + "_vgg_lstm")

        trainer = lightning.Trainer(
            devices=1,
            logger=logger,
            max_epochs=int(budget),
            # precision="bf16-mixed",
            check_val_every_n_epoch=100
        )

        trainer.fit(model=vgg_lstm, train_dataloaders=data_module.train_dataloader(batch_size=int(model_config['batch_size'])), val_dataloaders=data_module.val_dataloader(batch_size=int(model_config['batch_size'])))

        vgg_lstm_validation_metrics = trainer.validate(model=vgg_lstm, dataloaders=data_module.val_dataloader(batch_size=int(model_config['batch_size'])))
        
        vgg_lstm.cpu()
        del vgg_lstm
        
        model_config = self.top10nn_model_config.copy()
        model_config.update(config)
        model_config["dataset_name"] = data_module.get_dataset_name()

        top10nn = Top10NN(model_config)

        logger = TensorBoardLogger(save_dir="/home/malte.kuhlmann/youngs-modulus/data/logs", name=data_module.get_dataset_name() + "_top10nn")

        trainer = lightning.Trainer(
            devices=1,
            logger=logger,
            max_epochs=int(budget),
            # precision="bf16-mixed",
            check_val_every_n_epoch=100
        )

        trainer.fit(model=top10nn, train_dataloaders=data_module.train_dataloader(batch_size=int(model_config['batch_size'])), val_dataloaders=data_module.val_dataloader(batch_size=int(model_config['batch_size'])))

        top10nn_validation_metrics = trainer.validate(model=top10nn, dataloaders=data_module.val_dataloader(batch_size=int(model_config['batch_size'])))

        top10nn.cpu()
        del top10nn       

        data_module.full_dataset_file.unlink() 

        return {
            "val_n_mse": (res_tf_validation_metrics[0]['val_n_mse'] + vgg_lstm_validation_metrics[0]['val_n_mse'] + top10nn_validation_metrics[0]['val_n_mse']) / 3
        }

@hydra.main(version_base=None, config_path="/home/malte.kuhlmann/youngs-modulus/configs", config_name="optimization")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg["dataset"]["random_state"] = cfg["smac"]["random_state"]

    meta_data = OptimizationMetaData(cfg["dataset"], seed=cfg["smac"]["random_state"])

    scenario = Scenario(
        meta_data.configuration_space,
        name="2" + meta_data.get_name(),
        n_trials=cfg["smac"]["n_trials"],  
        deterministic=cfg["smac"]["deterministic"],  
        use_default_config=cfg["smac"]["use_default_config"], 
        # walltime_limit=cfg["smac"]["walltime_limit"],
        output_directory=cfg["smac"]["output_directory"],
        objectives=["val_n_mse"],
        min_budget=cfg["smac"]["min_budget"],
        max_budget=cfg["smac"]["max_budget"],
        n_workers=cfg["smac"]["n_workers"],
        seed=cfg["smac"]["random_state"]
    )

    smac = HyperparameterOptimizationFacade (
        scenario=scenario,
        target_function=meta_data.objective_function,
        overwrite=False
    )

    incumbent = smac.optimize()
    print("Best configuration:", incumbent)
    
if __name__ == "__main__":
    main()
