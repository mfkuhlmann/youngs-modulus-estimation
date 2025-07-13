import hydra
import lightning

from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from dataset.k_fold_youngs_data_module import KFoldYoungsDataModule
from model.Top10NN.modulus_model import ModulusModel

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac import MultiFidelityFacade, Scenario
#global_best_mse = np.inf
#global_best_config = None
#trial_counter = 0

class OptimizationMetaData():

    def __init__(self, dataset_cfg, seed=42):
        self.model_config = {
            # Dataset
            # Inside dataset_cfg
            
            #['playdoh', 'silly_puty', 'blue_sponge_dry', 'blue_sponge_wet', 'apple', 'orange', 'strawberry', 'ripe_banana', 'unripe_banana', 'lacrosse_ball', 'baseball', 'racquet_ball', 'tennis_ball'], 
            
            
            # Model Parameters
            'gamma': 0.975,
            'learning_rate': 0.00001,
            'lr_step_size': 1,
            'batch_size': 64,

            # Overall
            'random_state': 42, 
            
            # Model Architecture
            ## Dropouts
            'video_dropout': 0.1,
            'force_dropout': 0.1, # Not used
            'width_dropout': 0.1, # Not used
            'est_dropout': 0.1,
            'decoder_dropout': 0.1,

            ##Feature Sizes
            'img_feature_size': 128,
            'fwe_feature_size': 32, # Both force and width. Always one fully connected layer
            'decoder_size': [512, 512, 128], 
            'est_decoder_size': [64, 64, 32], 
            'decoder_output_size': 3, 
        }  

        self.model_config.update(dataset_cfg)

        self.seed = seed
        model_config = self.model_config
        self.data_module = KFoldYoungsDataModule(
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
            val_on_seen_objects=model_config["val_on_seen_objects"],
            use_log_normalization=model_config["use_log_normalization"],
            overwrite_file=False,
            batch_size=int(model_config['batch_size']),
            exclude=model_config['exclude'],
            random_state=42
        )

        self.data_module.prepare_data()
        self.data_module.setup()
    
    @property
    def configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        
        learning_rate = []
        for i in range(1, 10):  
            for j in range(2, 7):  
                lr = i * 10**(-j)
                learning_rate.append(lr)
                
        cs.add([
            CategoricalHyperparameter("learning_rate", learning_rate, default_value=0.00001),
            CategoricalHyperparameter("gamma", choices=[0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, None], default_value=0.975),
            CategoricalHyperparameter("batch_size", choices=[8, 16, 32, 64, 128], default_value=32),
            
            # Dropout
            CategoricalHyperparameter("video_dropout", choices=[0.0, 0.1, 0.2, 0.3, 0.4], default_value=0.0),
            CategoricalHyperparameter("decoder_dropout", choices=[0.0, 0.1, 0.2, 0.3, 0.4], default_value=0.0),
            CategoricalHyperparameter("est_dropout", choices=[0.0, 0.1, 0.2, 0.3, 0.4], default_value=0.0),

            # Weight and Force
            CategoricalHyperparameter("fwe_feature_size", choices=[8, 16, 32, 64, 128], default_value=32),

            # Transformer
            CategoricalHyperparameter("img_feature_size", choices=[32, 64, 128, 256, 512], default_value=128),

            # Estimation and Decoder
            CategoricalHyperparameter("decoder_size", choices=[[32, 32, 16], [64, 64, 32], [128, 128, 64], [256, 256, 128], [512, 512, 256]], default_value=[512, 512, 256]),
            CategoricalHyperparameter("est_decoder_size", choices=[[32, 32, 16], [64, 64, 32], [128, 128, 64], [256, 256, 128], [512, 512, 256]], default_value=[64, 64, 32]),
            CategoricalHyperparameter("decoder_output_size", choices=[1, 2, 3, 4, 5, 6, 7, 8], default_value=3),
        ])

        return cs
    
    def get_name(self):
        name = "Top10NN"

        if self.model_config["rubber_only"]:
            name += "_rubber_only"

        name += f"_{self.model_config['sample_type']}"
        if self.model_config["val_on_seen_objects"]:
            name += "_seen"
        else:
            name += "_unseen"
        
        if self.model_config["use_estimations"]:
            name += "_est"
        
        if self.model_config["use_force"]:
            name += "_force"
        
        if self.model_config["use_width"]:
            name += "_width"
        
        if self.model_config["use_width_transforms"]:
            name += "_width_transforms"

        if self.model_config["use_markers"]:
            name += "_only_markers"

        name += f"_random_state_{self.seed}"
        return name
    
    def objective_function(self, config: Configuration, seed=42, budget=10):
        
        config = dict(config)

        model_config = self.model_config.copy()

        model_config.update(config)
        
        model = ModulusModel(model_config)

        self.data_module.batch_size = int(model_config['batch_size'])
        
        logger = TensorBoardLogger(save_dir="/home/malte.kuhlmann/youngs-modulus/data/logs", name=self.get_name())

        trainer = lightning.Trainer(
            devices=1,
            logger=logger,
            max_epochs=int(budget),
            precision="bf16-mixed",
            check_val_every_n_epoch=100
        )

        trainer.fit(model=model, train_dataloaders=self.data_module.train_dataloader(), val_dataloaders=self.data_module.val_dataloader())

        validation_metrics = trainer.validate(model=model, dataloaders=self.data_module.val_dataloader())
        # del data_module
        return {
            "val_loss": validation_metrics[0]['val_loss'] 
        }

@hydra.main(version_base=None, config_path="/home/malte.kuhlmann/youngs-modulus/configs", config_name="optimization")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    meta_data = OptimizationMetaData(cfg["dataset"], seed=42)

    scenario = Scenario(
        meta_data.configuration_space,
        name=meta_data.get_name(),
        n_trials=cfg["smac"]["n_trials"],  
        deterministic=cfg["smac"]["deterministic"],  
        use_default_config=cfg["smac"]["use_default_config"], 
        # walltime_limit=cfg["smac"]["walltime_limit"],
        output_directory=cfg["smac"]["output_directory"],
        objectives=["val_loss"],
        min_budget=cfg["smac"]["min_budget"],
        max_budget=cfg["smac"]["max_budget"],
        n_workers=cfg["smac"]["n_workers"]
    )

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=meta_data.objective_function,
        overwrite=True
    )

    incumbent = smac.optimize()
    print("Best configuration:", incumbent)

if __name__ == "__main__":
    main()
    