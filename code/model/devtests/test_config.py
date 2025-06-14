from omegaconf import OmegaConf
from hydra import initialize, compose
import unittest


class TestEstimators(unittest.TestCase):
    def setUp(self):
        with initialize(version_base=None, config_path="../configs"):
            self.config = compose(
                config_name="train_config",
                overrides=[
                    "optimizer=sgd",
                    "features=[angles, distances]",
                    "general.log_path=papa",
                ],
            )
            OmegaConf.resolve(self.config)
            print(self.config)

    def test_config(self):
        assert isinstance(self.config.optimizer.params.lr, float)
        assert "class_name" in self.config.optimizer
        assert "class_name" in self.config.model
        assert isinstance(self.config.dataset.data_dir, str)
        assert isinstance(self.config.dataset.data_path, str)
        assert isinstance(self.config.training.num_epochs, int)
        assert isinstance(self.config.training.train_batch_size, int)
        assert isinstance(self.config.training.val_batch_size, int)
        assert isinstance(self.config.optimizer.params.momentum, float)
        assert (
            "cuda" in self.config.training.device
            or "cpu" in self.config.training.device
        )
        assert (
            self.config.training.type == "cross_validation"
            and isinstance(self.config.training.k_folds, int)
            or self.config.trainining.type != "cross_validation"
        )
