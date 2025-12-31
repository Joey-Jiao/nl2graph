from pathlib import Path

from ..base.configs import ConfigService
from ..seq2seq.train import ConfigLoader, Preprocessing, Training


class TrainPipeline:

    def __init__(self, config_service: ConfigService):
        self.config_service = config_service

    def preprocess(
        self,
        dataset_config_path: str,
        input_dir: Path,
        output_dir: Path,
    ):
        preprocessing = Preprocessing(self.config_service, dataset_config_path)
        preprocessing.process(input_dir, output_dir)

    def train(
        self,
        dataset_config_path: str,
        input_dir: Path,
        output_dir: Path,
        model_name_or_path: str,
    ) -> float:
        config_loader = ConfigLoader(dataset_config_path)
        dataset_config = config_loader.load()

        training = Training(
            config_service=self.config_service,
            dataset_config=dataset_config,
            input_dir=input_dir,
            output_dir=output_dir,
            model_name_or_path=model_name_or_path,
        )
        return training.train()
