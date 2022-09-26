import os
from io import BytesIO

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size
        train_files = [
            os.path.join(train_dir, file_name) for file_name in os.listdir(train_dir)
        ]
        self.train_dataset = wds.WebDataset(train_files).compose(load_tensors)
        val_files = [
            os.path.join(val_dir, file_name) for file_name in os.listdir(val_dir)
        ]
        self.val_dataset = wds.WebDataset(val_files).compose(load_tensors)

    def setup(self, stage=None):
        pass

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


def load_tensors(src):
    for sample in src:
        f = BytesIO(sample["pt"])
        t = torch.load(f)
        yield t
