import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import Dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        model_name: str,
        batch_size: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model_name = model_name

    def setup(self, stage=None):
        self.train_dataset = Dataset(
            self.train_dir,
            self.model_name,
            self.trainer.global_rank,
            self.trainer.world_size,
        )
        self.val_dataset = Dataset(
            self.val_dir,
            self.model_name,
            self.trainer.global_rank,
            self.trainer.world_size,
        )

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
