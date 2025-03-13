import hydra
from omegaconf import DictConfig
from rich import print


@hydra.main(config_path="./config", config_name="main", version_base="1.2")
def process_data(config: DictConfig):
    """Function to process the data"""

    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == "__main__":
    process_data()
