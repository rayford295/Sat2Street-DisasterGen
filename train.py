import hydra
from omegaconf import DictConfig
from src.data.datamodule import DataModule
from src.models.baseline import BaselineModel

@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # Simple training stub
    dm = DataModule(cfg.experiment.data_path)
    dm.setup()
    model = BaselineModel()
    print(f"Training {cfg.experiment.model} for {cfg.experiment.epochs} epochs")

if __name__ == "__main__":
    main()