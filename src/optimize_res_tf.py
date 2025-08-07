import hydra
import lightning

from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from dataset.k_fold_youngs_data_module import KFoldYoungsDataModule
from model.Res_TF.modulus_model import ModulusModel

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac import MultiFidelityFacade, Scenario

class OptimizationMetaData():

    def __init__(self, dataset_cfg, seed=42):
        self.model_config = {

            'gamma': 0.975,
            'learning_rate': 0.0001,


            # Model Architecture

            ## Dropout
            'video_dropout': 0.1,
            'est_dropout': 0.1,
            'decoder_dropout': 0.1,

            ## Feature Sizes
            'fwe_feature_size': 32, # Feature size for force, width. Only one fully connected layer
            'decoder_size': [512, 512, 128] ,
            'est_decoder_size': [64, 64, 32] ,
            'decoder_output_size': 3,

            ## Video Transformer
            'layer': 5,
            'head': 8,
            'embedding_dim':  512,
            'output_dim':  512,
            # Dataset
            # Inside dataset_cfg
            'batch_size': 32,
            
            #['playdoh', 'silly_puty', 'blue_sponge_dry', 'blue_sponge_wet', 'apple', 'orange', 'strawberry', 'ripe_banana', 'unripe_banana', 'lacrosse_ball', 'baseball', 'racquet_ball', 'tennis_ball'], 
            
            # Model Parameters

            'lr_step_size': 1,
            "use_fair_loss": False,

            # Model Architecture

            ## Dropout
            'force_dropout': 0.1, # Is not used
            'width_dropout': 0.1, # is not used
            
        }
        self.model_config.update(dataset_cfg)
        self.seed = seed

        model_config = self.model_config
        self.data_module = KFoldYoungsDataModule(
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
        self.model_config["dataset_name"] = self.data_module.get_dataset_name() # Add dataset name to model for later grouping
        self.model_config["model_name"] = "res_tf"

        self.data_module.prepare_data()
        self.data_module.setup()

    @property
    def configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        
        learning_rate = []
        for i in range(1, 10):  # 
            for j in range(2, 7):  # 
                lr = i * 10**(-j)
                learning_rate.append(lr)
                
        cs.add([
            CategoricalHyperparameter("learning_rate", learning_rate, default_value=0.00001),
            CategoricalHyperparameter("gamma", choices=[0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, None], default_value=0.975),
            CategoricalHyperparameter("batch_size", choices=[8, 16, 32, 64, 128], default_value=32),
            
            # Dropout
            CategoricalHyperparameter("video_dropout", choices=[0.0, 0.1, 0.2, 0.3, 0.4], default_value=0.0),
            CategoricalHyperparameter("decoder_dropout", choices=[0.0, 0.1, 0.2, 0.3, 0.4], default_value=0.0),
            

            # Weight and Force
            # CategoricalHyperparameter("fwe_feature_size", choices=[8, 16, 32, 64, 128], default_value=32), # Feature size for force, width. Only one fully connected layer

            # Transformer
            CategoricalHyperparameter("layer", choices=[2, 3, 4, 5, 6], default_value=5),
            CategoricalHyperparameter("head", choices=[1, 2, 4, 8, 16], default_value=8),
            CategoricalHyperparameter("embedding_dim", choices=[128, 256, 512, 1024], default_value=512),
            CategoricalHyperparameter("output_dim", choices=[128, 256, 512, 1024], default_value=512),

            # Estimation and Decoder
            CategoricalHyperparameter("decoder_size", choices=[[32, 32, 16], [64, 64, 32], [128, 128, 64], [256, 256, 128], [512, 512, 256]], default_value=[512, 512, 256]),
        ])

        if self.model_config["use_estimations"]:
            cs.add([
                CategoricalHyperparameter("est_decoder_size", choices=[[32, 32, 16], [64, 64, 32], [128, 128, 64], [256, 256, 128], [512, 512, 256]], default_value=[64, 64, 32]),
                CategoricalHyperparameter("decoder_output_size", choices=[1, 2, 3, 4, 5, 6, 7, 8], default_value=3),
                CategoricalHyperparameter("est_dropout", choices=[0.0, 0.1, 0.2, 0.3, 0.4], default_value=0.0),

            ])

        if self.model_config["use_force"] or self.model_config["use_width"]:
            cs.add([
                CategoricalHyperparameter("fwe_feature_size", choices=[8, 16, 32, 64, 128], default_value=32), # Feature size for force, width. Only one fully connected layer
            ])
        
        return cs
    
    def get_name(self):
        name = "Res_TF"

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
    
    def objective_function(self, config: Configuration, seed=0, budget=10):
        
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
            # precision="bf16-mixed",
            check_val_every_n_epoch=100
        )

        trainer.fit(model=model, train_dataloaders=self.data_module.train_dataloader(batch_size=int(model_config['batch_size'])), val_dataloaders=self.data_module.val_dataloader(batch_size=int(model_config['batch_size'])))

        validation_metrics = trainer.validate(model=model, dataloaders=self.data_module.val_dataloader(batch_size=int(model_config['batch_size'])))

        return {
            "val_n_mse": validation_metrics[0]['val_n_mse'] 
        }

@hydra.main(version_base=None, config_path="/home/malte.kuhlmann/youngs-modulus/configs", config_name="optimization")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg["dataset"]["random_state"] = cfg["smac"]["random_state"]

    meta_data = OptimizationMetaData(cfg["dataset"], seed=cfg["smac"]["random_state"])

    scenario = Scenario(
        meta_data.configuration_space,
        name=meta_data.get_name(),
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

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=meta_data.objective_function,
        overwrite=True
    )

    incumbent = smac.optimize()
    print("Best configuration:", incumbent)
    
if __name__ == "__main__":
    main()
